
import PIL
from PIL import Image

a ="C:\\Project\\integrate\\palm\\corpus\\src\\"
b ="C:\\Project\\integrate\\palm\\corpus\\des\\" 
for i in range (1,17):
	for j in range (1,11):
		basewidth = 700
		img_fn =a+str(i)+"\\"+str(j)+".jpg"
		img = Image.open(img_fn)
		wpercent = (basewidth / float(img.size[0]))
		hsize = int((float(img.size[1]) * float(wpercent)))
		img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
		img_sav =b+str(i)+"\\"+str(j)+".jpg" 
		img.save(img_sav)

