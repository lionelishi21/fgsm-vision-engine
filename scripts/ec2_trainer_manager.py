#!/usr/bin/env python3
"""
scripts/ec2_trainer_manager.py
EC2 GPU Training Manager for FGSM using the `aibridix_official` AWS Profile.

Supports:
- Provisioning cost-effective Spot/On-Demand GPU instances (g4dn.xlarge, g5.xlarge)
- Syncing training code & datasets via rsync
- Starting self-terminating training runs (shuts down automatically when done)
- Checking instance status & pulling trained model checkpoints
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import boto3

AWS_PROFILE = "aibridix_official"
AWS_REGION = "us-east-1"
DEFAULT_INSTANCE_TYPE = "g4dn.xlarge"  # NVIDIA T4 16GB VRAM (cheapest spot ~$0.16/hr)
# Alternative: "g5.xlarge" (NVIDIA A10G 24GB VRAM ~$0.30/hr spot)
INSTANCE_TAG = "fgsm-gpu-trainer"
KEY_NAME = "fgsm-trainer-key"
SG_NAME = "fgsm-trainer-sg"

# AWS Deep Learning OSS PyTorch AMI (Ubuntu 22.04) in us-east-1
DEFAULT_AMI_ID = "ami-012ba162b9cd2729c"


def get_session():
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def get_or_create_key_pair(ec2_client):
    ssh_dir = Path.home() / ".ssh"
    pub_key_path = ssh_dir / "id_ed25519.pub"
    if not pub_key_path.exists():
        pub_key_path = ssh_dir / "id_rsa.pub"

    try:
        ec2_client.describe_key_pairs(KeyNames=[KEY_NAME])
        return KEY_NAME
    except Exception:
        pass

    if pub_key_path.exists():
        with open(pub_key_path, "r") as f:
            pub_material = f.read().strip()
        ec2_client.import_key_pair(
            KeyName=KEY_NAME,
            PublicKeyMaterial=pub_material.encode("utf-8")
        )
        print(f"Imported public key from {pub_key_path} as '{KEY_NAME}'.")
        return KEY_NAME
    else:
        resp = ec2_client.create_key_pair(KeyName=KEY_NAME)
        pem_path = ssh_dir / f"{KEY_NAME}.pem"
        with open(pem_path, "w") as f:
            f.write(resp["KeyMaterial"])
        os.chmod(pem_path, 0o400)
        print(f"Created new key pair '{KEY_NAME}' and saved private key to {pem_path}.")
        return KEY_NAME


def get_or_create_security_group(ec2_client, vpc_id):
    try:
        sgs = ec2_client.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [SG_NAME]},
                {"Name": "vpc-id", "Values": [vpc_id]}
            ]
        )["SecurityGroups"]
        if sgs:
            return sgs[0]["GroupId"]
    except Exception:
        pass

    resp = ec2_client.create_security_group(
        GroupName=SG_NAME,
        Description="Security Group for FGSM GPU Trainer",
        VpcId=vpc_id
    )
    sg_id = resp["GroupId"]

    # Authorize SSH
    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}]
            }
        ]
    )
    print(f"Created security group '{SG_NAME}' ({sg_id}) allowing SSH.")
    return sg_id


def find_trainer_instance(ec2_client):
    resp = ec2_client.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [INSTANCE_TAG]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}
        ]
    )
    for res in resp.get("Reservations", []):
        for inst in res.get("Instances", []):
            return inst
    return None


def launch_instance(args):
    session = get_session()
    ec2_client = session.client("ec2")
    ec2_resource = session.resource("ec2")

    existing = find_trainer_instance(ec2_client)
    if existing:
        state = existing["State"]["Name"]
        inst_id = existing["InstanceId"]
        public_ip = existing.get("PublicIpAddress", "N/A")
        print(f"Found existing trainer instance {inst_id} in state: {state} (IP: {public_ip})")
        if state == "stopped":
            print(f"To start this instance, run: python scripts/ec2_trainer_manager.py start")
        return

    # Find default VPC
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        vpcs = ec2_client.describe_vpcs()["Vpcs"]
    vpc_id = vpcs[0]["VpcId"]

    key_name = get_or_create_key_pair(ec2_client)
    sg_id = get_or_create_security_group(ec2_client, vpc_id)

    instance_type = args.instance_type or DEFAULT_INSTANCE_TYPE
    spot = not args.on_demand

    print(f"Launching EC2 GPU Instance ({instance_type})...")
    print(f"  Spot Instance: {spot}")
    print(f"  AMI: {DEFAULT_AMI_ID}")
    print(f"  Region: {AWS_REGION} (Profile: {AWS_PROFILE})")

    instance_market_options = {
        "MarketType": "spot",
        "SpotOptions": {
            "SpotInstanceType": "one-time",
        }
    } if spot else None

    kwargs = {
        "ImageId": DEFAULT_AMI_ID,
        "InstanceType": instance_type,
        "KeyName": key_name,
        "SecurityGroupIds": [sg_id],
        "MinCount": 1,
        "MaxCount": 1,
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": args.volume_size,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True
                }
            }
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": INSTANCE_TAG}]
            }
        ],
        "IamInstanceProfile": {
            "Name": "fgsm-trainer-profile"
        }
    }

    if instance_market_options:
        kwargs["InstanceMarketOptions"] = instance_market_options

    res = ec2_client.run_instances(**kwargs)
    inst_id = res["Instances"][0]["InstanceId"]
    print(f"Instance requested: {inst_id}")
    print("Waiting for instance to enter 'running' state...")

    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[inst_id])

    inst_info = ec2_client.describe_instances(InstanceIds=[inst_id])["Reservations"][0]["Instances"][0]
    public_ip = inst_info.get("PublicIpAddress")
    print(f"\n=======================================================")
    print(f" Trainer Instance Ready!")
    print(f" ID:        {inst_id}")
    print(f" State:     {inst_info['State']['Name']}")
    print(f" Public IP: {public_ip}")
    print(f" SSH:       ssh -i ~/.ssh/{key_name} ubuntu@{public_ip}")
    print(f"=======================================================\n")
    print(f"Next steps:")
    print(f"  1. Push code & data:  python scripts/ec2_trainer_manager.py push")
    print(f"  2. Start training:    python scripts/ec2_trainer_manager.py start-training")


def show_status(args):
    session = get_session()
    ec2_client = session.client("ec2")
    inst = find_trainer_instance(ec2_client)
    if not inst:
        print("No FGSM trainer instance found.")
        return

    inst_id = inst["InstanceId"]
    state = inst["State"]["Name"]
    inst_type = inst["InstanceType"]
    public_ip = inst.get("PublicIpAddress", "None (Instance stopped)")
    launch_time = inst.get("LaunchTime")

    print(f"Instance ID:    {inst_id}")
    print(f"State:          {state}")
    print(f"Type:           {inst_type}")
    print(f"Public IP:      {public_ip}")
    print(f"Launch Time:    {launch_time}")


def start_instance(args):
    session = get_session()
    ec2_client = session.client("ec2")
    inst = find_trainer_instance(ec2_client)
    if not inst:
        print("No instance found.")
        return
    inst_id = inst["InstanceId"]
    print(f"Starting instance {inst_id}...")
    ec2_client.start_instances(InstanceIds=[inst_id])
    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[inst_id])
    updated = ec2_client.describe_instances(InstanceIds=[inst_id])["Reservations"][0]["Instances"][0]
    print(f"Instance {inst_id} running at IP: {updated.get('PublicIpAddress')}")


def stop_instance(args):
    session = get_session()
    ec2_client = session.client("ec2")
    inst = find_trainer_instance(ec2_client)
    if not inst:
        print("No instance found.")
        return
    inst_id = inst["InstanceId"]
    print(f"Stopping instance {inst_id}...")
    ec2_client.stop_instances(InstanceIds=[inst_id])
    print(f"Instance {inst_id} stopping. Billing for compute has stopped.")


def terminate_instance(args):
    session = get_session()
    ec2_client = session.client("ec2")
    inst = find_trainer_instance(ec2_client)
    if not inst:
        print("No instance found.")
        return
    inst_id = inst["InstanceId"]
    confirm = input(f"Are you sure you want to TERMINATE {inst_id}? (y/N): ")
    if confirm.lower() == "y":
        ec2_client.terminate_instances(InstanceIds=[inst_id])
        print(f"Instance {inst_id} terminated.")


def push_code_and_data(args):
    session = get_session()
    ec2_client = session.client("ec2")
    inst = find_trainer_instance(ec2_client)
    if not inst or inst["State"]["Name"] != "running":
        print("Trainer instance is not running. Start or launch it first.")
        return

    public_ip = inst["PublicIpAddress"]
    print(f"Syncing codebase and dataset to ubuntu@{public_ip}:/home/ubuntu/fgsm-vision-engine/ ...")

    # Ensure remote directory exists
    subprocess.run([
        "ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{public_ip}",
        "mkdir -p /home/ubuntu/fgsm-vision-engine"
    ], check=True)

    # Rsync repo files excluding venv, caches, and stale local checkpoints.
    # --partial + a keepalived ssh transport: this repo's dataset pushes run
    # several GB / several minutes, and plain rsync over ssh was getting
    # dropped ("Broken pipe") mid-transfer without server-side keepalives.
    rsync_cmd = [
        "rsync", "-avz", "--progress", "--partial",
        "-e", "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10",
        "--exclude", ".venv",
        "--exclude", ".git",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", "runs/",
        "./", f"ubuntu@{public_ip}:/home/ubuntu/fgsm-vision-engine/"
    ]
    subprocess.run(rsync_cmd, check=True)
    print("\nSync completed successfully!")


def start_training(args):
    session = get_session()
    ec2_client = session.client("ec2")
    inst = find_trainer_instance(ec2_client)
    if not inst or inst["State"]["Name"] != "running":
        print("Trainer instance is not running.")
        return

    public_ip = inst["PublicIpAddress"]
    print(f"Launching self-terminating training job on ubuntu@{public_ip}...")

    cmd = (
        "cd /home/ubuntu/fgsm-vision-engine && "
        "chmod +x scripts/train_and_stop.sh && "
        "nohup bash scripts/train_and_stop.sh > training_job.log 2>&1 & "
        "echo 'Training job started in background!'"
    )
    subprocess.run([
        "ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{public_ip}", cmd
    ], check=True)

    print("\n=======================================================")
    print(" Job started! When training completes, the instance will")
    print(" automatically shut down to prevent any idle AWS charges.")
    print(f" View live logs anytime with:")
    print(f"   ssh ubuntu@{public_ip} 'tail -f /home/ubuntu/fgsm-vision-engine/training_job.log'")
    print("=======================================================\n")


def check_quota(args):
    session = get_session()
    sq_client = session.client("service-quotas")
    
    quotas = [
        ("L-DB2E81BA", "Running On-Demand G and VT instances"),
        ("L-3819A6DF", "All G and VT Spot Instance Requests"),
    ]
    
    print("=== AWS EC2 GPU Quota Status (us-east-1) ===")
    for code, name in quotas:
        try:
            q = sq_client.get_service_quota(ServiceCode="ec2", QuotaCode=code)["Quota"]
            val = q.get("Value", 0)
            print(f"  {name}: {val} vCPUs")
        except Exception as e:
            print(f"  {name}: Error retrieving ({e})")
            
    # Check pending requests
    try:
        reqs = sq_client.list_requested_service_quota_change_history_by_quota(
            ServiceCode="ec2", QuotaCode="L-DB2E81BA"
        ).get("RequestedQuotas", [])
        for r in reqs:
            print(f"  On-Demand Request Status: {r['Status']} (Desired: {r['DesiredValue']} vCPUs)")
    except Exception:
        pass


def pull_weights(args):
    print("Pulling trained models from S3 bucket (s3://fgsm-vision-models-aibridix-official/runs/) to ./runs/ ...")
    os.makedirs("./runs", exist_ok=True)
    sync_cmd = [
        "aws", "s3", "sync",
        "s3://fgsm-vision-models-aibridix-official/runs/", "./runs/",
        "--profile", AWS_PROFILE,
        "--region", AWS_REGION
    ]
    subprocess.run(sync_cmd, check=True)
    print("\nDownloaded checkpoints successfully!")


def main():
    parser = argparse.ArgumentParser(description="FGSM EC2 GPU Trainer Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Launch
    p_launch = subparsers.add_parser("launch", help="Launch EC2 GPU Spot instance")
    p_launch.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="e.g. g4dn.xlarge, g5.xlarge")
    p_launch.add_argument("--volume-size", type=int, default=150, help="Root EBS size in GB")
    p_launch.add_argument("--on-demand", action="store_true", help="Use On-Demand instead of Spot")
    p_launch.set_defaults(func=launch_instance)

    # Status
    p_status = subparsers.add_parser("status", help="Show trainer instance status")
    p_status.set_defaults(func=show_status)

    # Quota
    p_quota = subparsers.add_parser("check-quota", help="Check AWS EC2 GPU vCPU quotas and request status")
    p_quota.set_defaults(func=check_quota)

    # Start / Stop / Terminate
    p_start = subparsers.add_parser("start", help="Start existing stopped trainer instance")
    p_start.set_defaults(func=start_instance)

    p_stop = subparsers.add_parser("stop", help="Stop running instance")
    p_stop.set_defaults(func=stop_instance)

    p_term = subparsers.add_parser("terminate", help="Terminate instance permanently")
    p_term.set_defaults(func=terminate_instance)

    # Push / Train / Pull
    p_push = subparsers.add_parser("push", help="Rsync code and datasets to remote instance")
    p_push.set_defaults(func=push_code_and_data)

    p_train = subparsers.add_parser("start-training", help="Launch background training with auto-shutdown")
    p_train.set_defaults(func=start_training)

    p_pull = subparsers.add_parser("pull", help="Pull trained weights and runs back to local")
    p_pull.set_defaults(func=pull_weights)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
