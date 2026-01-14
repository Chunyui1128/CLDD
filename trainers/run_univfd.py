# trainers/run_univfd.py
import os
import subprocess

def main():
    """
    Run a smoke test of UnivFD on CPU.
    Adjust paths and options as needed.
    """
    # 数据列表路径
    real_list = "datasets/real_train.pickle"
    fake_list = "datasets/fake_train.pickle"

    # smoke test 参数
    batch_size = 2
    niter = 1
    arch = "Imagenet:resnet50"
    experiment_name = "smoke_test"

    # 构造命令
    cmd = [
        "python", "-m", "methods.univfd.train",
        "--data_mode", "ours",
        "--real_list_path", real_list,
        "--fake_list_path", fake_list,
        "--batch_size", str(batch_size),
        "--niter", str(niter),
        "--gpu_ids", "",                # 空字符串表示使用 CPU
        "--arch", arch,
        "--name", experiment_name,
        "--fix_backbone"                # 避免提示
    ]

    print("Running UnivFD smoke test...")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

