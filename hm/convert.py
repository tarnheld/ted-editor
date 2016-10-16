from PIL import Image
with open("Andalusia.raw", mode='rb') as file:
    raw = file.read()
    img = Image.frombytes("F",(1024,1024),raw,"raw","F;16")
    #f = 1.0 / 2**16
    #img = img.point(lambda x: x * f)
    img = img.convert("I;16")
    img.save("Andalusia.tif","TIFF")

