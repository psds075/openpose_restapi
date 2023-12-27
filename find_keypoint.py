import os
import torch
import cv2
import numpy as np
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
import openpose


def prediction(img_path = "test.png", height_size = 512):

	pose_dict = {
		0 : "얼굴",
		1 : "어깨 중심",
		2 : "어깨 좌측",
		3 : "팔 좌측",
		4 : "손목 좌측",
		5 : "어깨 우측",
		6 : "팔 우측",
		7 : "손목 우측",
		8 : "좌측 골반",
		9 : "좌측 무릎",
		10 : "좌측 발",
		11 : "우측 골반",
		12 : "우측 무릎",
		13 : "우측 발",
		14 : "좌측 눈",
		15 : "우측 눈",
		16 : "좌측 귀",
		17 : "우측 귀"}

	net = PoseEstimationWithMobileNet()
	checkpoint = torch.load('weights/openpose.pth', map_location='cpu')
	load_state(net, checkpoint)

	net = net.cuda()
	net = net.eval()
	stride = 8
	upsample_ratio = 4
	num_keypoints = Pose.num_kpts
	previous_poses = []
	delay = 33

	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	orig_img = img.copy()
	orig_img = img.copy()
	heatmaps, pafs, scale, pad = openpose.infer_fast(net, img, height_size, stride, upsample_ratio, cpu=False)

	total_keypoints_num = 0
	all_keypoints_by_type = []
	for kpt_idx in range(num_keypoints):  # 19th for bg
		total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

	pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
	for kpt_id in range(all_keypoints.shape[0]):
		all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
		all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

	result = []
	for n in range(len(pose_entries)):
		if len(pose_entries[n]) == 0:
			continue
		pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
		valid_keypoints = []
		for kpt_id in range(num_keypoints):
			if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
				pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
				pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
				valid_keypoints.append([pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1]])
		valid_keypoints = np.array(valid_keypoints)
		result.append(valid_keypoints)

	my_keypoints = []
	for idx, keypoint in enumerate(result[0]):
		my_keypoints.append([idx, pose_dict[idx],int(keypoint[0]), int(keypoint[1])])

	return my_keypoints

def draw_point(image, x, y, point_color=(0, 0, 255), point_radius=3):
    cv2.circle(image, (x, y), point_radius, point_color, -1)
    return image


if __name__ == "__main__":
    
	all_keypoints = prediction(img_path = "test.png", height_size = 512)

	import json
	with open("keypoint.json", "w") as f:
		json_data = json.dumps(all_keypoints, ensure_ascii=False)
		f.write(json_data)

	# write result
	image = cv2.imread("test.png")
	for idx, keypoint in enumerate(all_keypoints):
		image = draw_point(image, int(keypoint[2]), int(keypoint[3]))

	cv2.imwrite("pointout.png",image)


	



