# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:27:37 2018

@author: mtroyer
"""

import os
import csv
import PIL.Image
import PIL.ExifTags


def convert_to_degress(value):
    """
    Convert exif GPS coordinate tuples to decimal degress
    """
    d = float(value[0][0]) / float(value[0][1])
    m = float(value[1][0]) / float(value[1][1])
    s = float(value[2][0]) / float(value[2][1])

    return d + (m / 60.0) + (s / 3600.0)


def getCoords(filepath):

    img = PIL.Image.open(filepath)

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in img._getexif().items()
        if k in PIL.ExifTags.TAGS
    }
    
    gpsinfo = {}
    for key in exif['GPSInfo'].keys():
        decode = PIL.ExifTags.GPSTAGS.get(key,key)
        gpsinfo[decode] = exif['GPSInfo'][key]

    latitude = gpsinfo['GPSLatitude']
    latitude_ref = gpsinfo['GPSLatitudeRef']
    lat_value = convert_to_degress(latitude)
    if latitude_ref == u'S':
        lat_value = -lat_value
                
    longitude = gpsinfo['GPSLongitude']
    longitude_ref = gpsinfo['GPSLongitudeRef']
    lon_value = convert_to_degress(longitude)
    if longitude_ref == 'W':
        lon_value = -lon_value

    return {'latitude': lat_value, 'longitude': lon_value}


def picsToCoordCSV(folder):
    pic_formats = ('.png', '.jpeg', '.jpg')
    
    pics = [f for f in os.listdir(folder) if os.path.splitext(f)[1] in pic_formats]
        
    coords = {}
    for pic in pics:
        try:
            coords[pic] = getCoords(os.path.join(folder, pic))
        except:
            pass


    with open(os.path.join(folder, 'coords.csv'), 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['name', 'path', 'latitude', 'longitude'])
        for name, c in coords.items():
            row = (name, os.path.join(folder, name), c['latitude'], c['longitude'])
            csvwriter.writerow(row)
                
                
if __name__ == '__main__':
    pic_path = r'..\Documents\pic2.jpg'
    gps = getCoords(pic_path)
    print gps
