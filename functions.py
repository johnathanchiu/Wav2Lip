from audio import *
import face_detection

import numpy as np
import torch
import cv2

from tqdm import tqdm
import subprocess


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def face_detect(images, max_frames=25, batch_size=1, pads=(0,10,0,0), nosmooth=False):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, min(len(images), max_frames), batch_size)):
				predictions.append(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = pads
	for frame_pred in predictions:
		frame_results = []
		for rect, image in zip(frame_pred, images):
			if rect is None:
				cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
				raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

			y1 = max(0, rect[1] - pady1)
			y2 = min(image.shape[0], rect[3] + pady2)
			x1 = max(0, rect[0] - padx1)
			x2 = min(image.shape[1], rect[2] + padx2)

			frame_results.append([x1, y1, x2, y2])
			
		results.append(frame_results)

	boxes = np.array(results)
	if not nosmooth: 
		boxes = get_smoothened_boxes(boxes, T=5)

	final_results = []
	for i, image in enumerate(images[:len(boxes)]):
		curr_results = []
		for (x1, y1, x2, y2) in boxes[i]:
			curr_results.append([image[y1: y2, x1:x2], (y1, y2, x1, x2)])
		final_results.append(curr_results)

	del detector
	return final_results 


def read_frames_from_file(vid_file, crop=(0, -1, 0, -1), resize_factor=1, rotate=False):
	video_stream = cv2.VideoCapture(vid_file)
	fps = video_stream.get(cv2.CAP_PROP_FPS)

	print('Reading video frames...')

	full_frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		if resize_factor > 1:
			frame = cv2.resize(frame, (frame.shape[1]//reesize_factor, frame.shape[0]//resize_factor))

		if rotate:
			frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

		y1, y2, x1, x2 = crop
		if x2 == -1: x2 = frame.shape[1]
		if y2 == -1: y2 = frame.shape[0]

		frame = frame[y1:y2, x1:x2]

		full_frames.append(frame)

	return full_frames


def compute_mel_spectrogram(audio_file, fps=25, mel_step_size=16, sr=16000):
	if not audio_file.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_file, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		audio_file = 'temp/temp.wav'

	wav = load_wav(audio_file, sr)
	mel = melspectrogram(wav)

	return mel


def fix_faces_to_idx(faces):
	num_faces = max([len(f) for f in faces])
	face_dict = {i: [] for i in range(num_faces)}
	for i in range(len(faces)):
		frame_info = faces[idx]
		for j in range(num_faces):
			frame_info
			
	return face_dict