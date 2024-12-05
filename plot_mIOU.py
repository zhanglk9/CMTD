import json
import matplotlib.pyplot as plt

# Load the log file to extract mIoU and corresponding iterations
file_path1 = "/home/dushiyan/repositories/HRDA/work_dirs/local-basic/241201_0409_gtaHR2csHR_hrda_s1_beit_adapter_2_6febf/20241201_040932.log.json"
file_path2 = "/home/dushiyan/repositories/HRDA/work_dirs/local-basic/Disable_fd_80000_iters_2e-05/20241201_210624.log.json"
file_path3 = "/home/dushiyan/repositories/HRDA/work_dirs/local-basic/Disable_fd_beitadapter_reimplmentation/20241130_083938.log.json"
file_path4 = "/home/dushiyan/repositories/HRDA/work_dirs/local-basic/Disable_fd_reimplementation/20241130_083227.log.json"

# Read the JSON content
def read_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                # Parse each JSON line
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Extract mIoU and iteration information
    iterations = []
    mIoU_values = []
    for i, entry in enumerate(data):
        if "mIoU" in entry:  # Assuming "decode.acc_seg" represents mIoU (accuracy of segmentation)
            iters = data[i - 1].get("iter", 0) + 50
            iterations.append(iters)
            mIoU_values.append(entry["mIoU"] * 100)
    return iterations, mIoU_values


it1, miou1 = read_json(file_path1)
it2, miou2 = read_json(file_path2)
it3, miou3 = read_json(file_path3)
it4, miou4 = read_json(file_path4)


# Plotting the mIoU curve
plt.figure(figsize=(10, 6))
plt.plot(it1, miou1, marker='o', label='backbone:Beit_adapter.lr:2e-05;iters:80000')
plt.plot(it3, miou3, marker='o', label='backbone:Beit_adapter.lr:6e-05;iters:40000')
plt.plot(it2, miou2, marker='o', label='backbone:mitb5.lr:2e-05;iters:80000')
plt.plot(it4, miou4, marker='o', label='backbone:mitb5.lr:6e-05;iters:40000')
plt.xlabel('Iterations')
plt.ylabel('mIoU')
plt.title('mIoU over Iterations without FD')
plt.grid()
plt.legend()
plt.savefig('mIOU.jpg')
