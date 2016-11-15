# code by Razerman, thanks!
import requests
import json
import zlib
import struct
import os

# Helper function to strip out bytes
def remove_bytes(buffer, start, end):
    fmt = '%ds %dx %ds' % (start, end - start, len(buffer) - end)  # 3 way split
    return b''.join(struct.unpack(fmt, buffer))

# Helper function to get byte out of int
def byte(number, i):
    return (number & (0xff << (i * 8))) >> (i * 8)

# Function to encrypt and decrypt GT6TED
def xor(data):
    TedKey = bytes([0x45, 0x32, 0x35, 0x67, 0x65, 0x69, 0x72, 0x45, 0x50, 0x48, 0x70, 0x63, 0x34, 0x57, 0x47, 0x32, 0x46, 0x6E, 0x7A, 0x61, 0x63, 0x4D, 0x71, 0x72, 0x75])
    return (XorEncript(data, TedKey))

# Does the actual encryption/decryption for xor function
def XorEncript(data, key):
    buffer = [None] * len(data)
    index = 0
    for i in range(len(data)):
        if(index < len(key)):
            index += 1
        else:
            index = 1
        buffer[i] = (data[i] ^ key[index - 1])
    return bytes(buffer)

# Compresses the GT6TED, adds the magic and negated file length and encrypts
# the data
def deflate(data, compresslevel=9):
    # Compress
    compress = zlib.compressobj(compresslevel, zlib.DEFLATED, -zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, 0)
    deflated = compress.compress(data)
    deflated += compress.flush()
    # Add PDs compression magic and negated file length (8 bytes)
    length = int(-len(deflated))
    magic = bytearray(b'\xc5\xee\xf7\xff\x00\x00\x00\x00')
    magic[4] = byte(length, 0)
    magic[5] = byte(length >> 8, 0)
    magic[6] = byte(length >> 0x10, 0)
    magic[7] = byte(length >> 0x18, 0)
    deflatedwithheader = bytes(magic) + deflated
    finaldata = xor(deflatedwithheader)
    return finaldata

# Decrypts the GT6TED, removes the magic and negated file length and
# decompresses the data
def inflate(data):
    decrypted = xor(data)
    headerstripped = remove_bytes(decrypted, 0, 8)
    decompress = zlib.decompressobj(-zlib.MAX_WBITS)
    inflated = decompress.decompress(headerstripped)
    inflated += decompress.flush()
    return inflated

def checkTedDataValidity(data):
    if data[0:6] == b'GT6TED':
        return deflate(data, 6)
    elif data[0:4] == b'\x80\xdc\xc2\x98':
        return data
    else:
        return None
# Checks if the file is unencrypted -> encrypts, encrypted -> all good or invalid file -> None
def checkTedFileValidity(filepath):
    myted = open(filepath, 'rb').read()
    return checkTedDataValidity(myted)
# Checks if the cookie is valid, returns username if valid, else None
def checkCookieValidity(cookie):
    headers = {'Cookie': 'JSESSIONID=' + cookie}
    r = requests.post('http://www.gran-turismo.com/gb/ajax/get_online_id/', headers=headers)
    username = json.loads(r.text)['online_id']
    if username:
        return username
    else:
        return None
# Uploads the ted file to PD servers (Does have a checking for Cookie validity)
def uploadTedData(data, title, cookie, username):
    headers = {'Cookie': 'JSESSIONID=' + cookie}
    files = {'data': ('gt6.ted', data)}
    data = {'job': (None, '1'), 'user_id': (None, username), 'title': (None, title)}
    res = requests.post('https://www.gran-turismo.com/jp/api/gt6/course/', files=files, data=data, headers=headers, verify=False)
    uploadResult = json.loads(res.text)['result']
    if uploadResult == 1:
        return True
    else:
        return False
def uploadTedFile(filepath, title, cookie):
    data    = checkTedFileValidity(filepath)
    username = checkCookieValidity(cookie)
    if data and username:
        return uploadTedData(data,title,cookie,username)
    else:
        return False
def uploadData(data, title, cookie):
    data = checkDataValidity(data)
    username = checkCookieValidity(cookie)
    if data and username:
        return uploadTedData(data,title,cookie,username)
    else:
        return False

if __name__ == "__main__":
    # Call Upload
    uploadTedFile("test.ted", "Python Upload Test", "JSESSIONID=")

    # Encrypt tests
    filepath = "test.ted"
    filename, file_extension = os.path.splitext(filepath)
    data = open(filepath, 'rb').read()
    encrypted = deflate(data, 6)
    f = open(filename + "_encrypted.ted", 'wb')
    f.write(encrypted)
    f.close()

    # Decrypt tests
    encrypted = open(filename + "_encrypted.ted", 'rb').read()
    decrypted = inflate(encrypted)
    b = open(filename + "_decrypted.ted", 'wb')
    b.write(decrypted)
    b.close()
