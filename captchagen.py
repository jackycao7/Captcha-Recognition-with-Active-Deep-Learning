import json
import string
import os
import uuid
from captcha.image import ImageCaptcha
import random
import shutil

def createCaptchas(dataDir, epoch, width, height, domain, train = True):
	if os.path.exists(dataDir):
		shutil.rmtree(dataDir)
	if not os.path.exists(dataDir):
		os.makedirs(dataDir)

	captchaImage = ImageCaptcha(width=width, height=height)

	if(train):
		print('generating training data')
	else:
		print('generating test data')
	for x in range(epoch):
		if(train):
			totalImages = 30000
		else:
			totalImages = 10000
		for i in range(totalImages):
			newCaptcha = ''.join(random.sample(domain,4))
			newFile = os.path.join(dataDir, '%s_%s.png' % (newCaptcha, uuid.uuid4()))
			captchaImage.write(newCaptcha, newFile)



createCaptchas(os.path.join("./data4len4digi/train"), epoch = 1, width = 120, height = 100, 
	domain = "0123456789", train = True)

createCaptchas(os.path.join("./data4len4digi/test"), epoch = 1, width = 120, height = 100, 
	domain = "0123456789", train = False)
