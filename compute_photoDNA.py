#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import importlib
from ctypes import cast
from ctypes import cdll
from ctypes import c_int
from ctypes import c_ubyte
from ctypes import POINTER
from ctypes import c_char_p

import numpy as np

try:
	from PIL import Image, ImageFile
	ImageFile.LOAD_TRUNCATED_IMAGES = True
except:
	print('Install Pillow with "pip3 install Pillow"')
	quit()

inputFolder = r'C:\Users\youss\Desktop\Uni\Master\S3\lab-visual-computing\CertPhash\train_verify\data\test'


def _default_library_name():
	if sys.platform == "win32":
		return 'PhotoDNAx64.dll'
	if sys.platform == "darwin":
		return 'PhotoDNAx64.so'
	raise OSError('Linux is not supported.')


def load_stl10_dataset(root='./data', split='train', download=True, transform=None):
	try:
		stl10_module = importlib.import_module('torchvision.datasets')
	except Exception:
		raise ImportError('Install torchvision to load STL10 datasets.')
	return stl10_module.STL10(root=root, split=split, download=download, transform=transform)


def _to_pil_image(sample):
	if isinstance(sample, Image.Image):
		image = sample
	else:
		if hasattr(sample, 'detach'):
			sample = sample.detach().cpu().numpy()
		else:
			sample = np.asarray(sample)

		if sample.ndim == 3 and sample.shape[0] in (1, 3):
			sample = np.transpose(sample, (1, 2, 0))

		if sample.dtype != np.uint8:
			if np.issubdtype(sample.dtype, np.floating):
				sample = np.clip(sample * 255.0, 0, 255)
			else:
				sample = np.clip(sample, 0, 255)
			sample = sample.astype(np.uint8)

		image = Image.fromarray(sample)

	if image.mode != 'RGB':
		image = image.convert(mode='RGB')
	return image


def _compute_hash_array(imageFile, outputFolder=None, libName=None):
	try:
		if outputFolder is None:
			outputFolder = os.getcwd()
		if libName is None:
			libName = _default_library_name()

		if imageFile.mode != 'RGB':
			imageFile = imageFile.convert(mode='RGB')
		libPhotoDNA = cdll.LoadLibrary(os.path.join(outputFolder, libName))

		ComputeRobustHash = libPhotoDNA.ComputeRobustHash
		ComputeRobustHash.argtypes = [c_char_p, c_int, c_int, c_int, POINTER(c_ubyte), c_int]
		ComputeRobustHash.restype = c_ubyte

		hashByteArray = (c_ubyte * 144)()
		ComputeRobustHash(c_char_p(imageFile.tobytes()), imageFile.width, imageFile.height, 0, hashByteArray, 0)

		hashPtr = cast(hashByteArray, POINTER(c_ubyte))
		hashArray = np.frombuffer(bytes(hashPtr[:144]), dtype=np.uint8).copy()
		return hashArray
	except Exception as e:
		print(e)
		return np.array([], dtype=np.uint8)


def compute(image, outputFolder=None, libName=None):
	"""Compute a single PhotoDNA hash from one image.

	Accepts either a PIL image or a NumPy image array like `img_np`.
	Returns a tuple `(hash_vec, quality)` to mirror the PDQ call site.
	"""
	imageFile = _to_pil_image(image)
	hashArray = _compute_hash_array(imageFile, outputFolder=outputFolder, libName=libName)
	return hashArray.astype(np.float32), 1.0


def generate_hashes(source, output_csv=None, outputFolder=None, libName=None):
	"""Generate PhotoDNA hashes from either a dataset or an input directory.

	If source is a directory path, the function saves a CSV with image paths and hashes
	and returns the hash array. If source is a dataset-like iterable, the function returns
	the hashes directly without saving a file.
	"""
	if outputFolder is None:
		outputFolder = os.getcwd()
	if libName is None:
		libName = _default_library_name()

	hashes = []
	rows = []

	if isinstance(source, str):
		images = glob.glob(os.path.join(source, '**', '*.jp*g'), recursive=True)
		images.extend(glob.glob(os.path.join(source, '**', '*.png'), recursive=True))
		images.extend(glob.glob(os.path.join(source, '**', '*.gif'), recursive=True))
		images.extend(glob.glob(os.path.join(source, '**', '*.bmp'), recursive=True))

		for imagePath in images:
			with Image.open(imagePath, 'r') as imageFile:
				hashArray = _compute_hash_array(imageFile, outputFolder=outputFolder, libName=libName)
			if hashArray.size:
				hashes.append(hashArray)
				rows.append((imagePath, ','.join(map(str, hashArray.tolist()))))

		if output_csv is None:
			output_csv = os.path.join(source, 'image_hash.csv')

		with open(output_csv, 'w', encoding='utf-8', newline='') as file:
			for imagePath, hashString in rows:
				file.write(f'"{imagePath}","{hashString}"\n')

		return np.asarray(hashes, dtype=np.uint8)

	for sample in source:
		image = sample[0] if isinstance(sample, tuple) else sample
		image = _to_pil_image(image)
		hashArray = _compute_hash_array(image, outputFolder=outputFolder, libName=libName)
		if hashArray.size:
			hashes.append(hashArray)

	return np.asarray(hashes, dtype=np.uint8)

if __name__ == '__main__':
	try:
		libName = _default_library_name()
	except OSError as exc:
		print(exc)
		quit()
	use_stl10_dataset = False
	if use_stl10_dataset:
		dataset = load_stl10_dataset(root='./data', split='train', download=True, transform=None)
		hashes = generate_hashes(dataset, outputFolder=os.getcwd(), libName=libName)
	else:
		hashes = generate_hashes(inputFolder, output_csv=os.path.join(os.getcwd(), 'image_hash.csv'), outputFolder=os.getcwd(), libName=libName)
	print('Generated ' + f'{len(hashes):,}' + ' hashes.')
	print(hashes)