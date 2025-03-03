import requests
import googlemaps
import os
import pandas as pd
import random
import logging
# import minio API package


# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # handlers=[
    #     logging.StreamHandler(),
    #     logging.FileHandler('image_clipper.log')  #
    # ]
)
logger = logging.getLogger(__name__)

class GoogleSVI_store:
    def __init__(self, api_key, parent_folder):
        """Initialize the Google Street View Image (GSVI) uploader."""
        self.api_key = api_key
        self.gmaps = googlemaps.Client(key=api_key)
        self.parent_folder = parent_folder
        self.image_cache_path = os.path.join(self.parent_folder, "cache/images/")
        os.makedirs(self.image_cache_path, exist_ok=True)

    def pull_image(self, address, city_count,size="640x640", name=""):
        """Fetches an image from the Google Maps Street View API for a given address."""
        pic_base = 'https://maps.googleapis.com/maps/api/streetview?'
        pic_params = {
            'key': self.api_key,
            'location': address,
            'size': size #"640x640"
        }

        try:
            response = requests.get(pic_base, params=pic_params)
            response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
            image_name = f"{name}_{city_count}.png"
            image_path = os.path.join(self.image_cache_path, image_name)

            # save image locally
            with open(image_path, "wb") as file:
                file.write(response.content)

            # Save metadata of an image


            print(f"Image saved: {image_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image for {address}: {e}")
        finally:
            response.close()

    def select_pano_ids(self, geom):
        """Select all pano_ids and their attributes"""
        return None

if __name__ == "__main__":
    import requests
    # using matpltolib to display the image
    import matplotlib
    import json
    matplotlib.use('Agg')#TkAgg
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg


    meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
    pic_base = 'https://maps.googleapis.com/maps/api/streetview?'

    api_key = 'AIzaSyCf-gm_UHKwm3Rk-Qok1LjIZ-jrVmhGvps'
    location = '49.482754476094,8.479343325212051'
    # define the params for the metadata reques
    meta_params = {'key': api_key,
                   'location': location}
    # define the params for the metadat from elevationAPI
    meta_eparams = {'key': api_key,
                    'locations': location}
    # define the params for the picture request
    pic_params = {'key': api_key,
                  'location': location,
                  'heading':0,
                  'pitch':0,
                  'size': "640x640"}
    # obtain the metadata of the request (this is free)
    meta_response = requests.get(meta_base, params=meta_params)
    # display the contents of the response
    # the returned value are in JSON format
    meta_response.json()
    pic_meta = meta_response.content
    decoded_string = pic_meta.decode('utf-8')
    # Convert the string into a Python dictionary
    json_data = json.loads(decoded_string)

    pic_response = requests.get(pic_base, params=pic_params)

    for key, value in pic_response.headers.items():
        print(f"{key}: {value}")

    with open('cache/images/test3.jpg', 'wb') as file:
        file.write(pic_response.content)
    # remember to close the response connection to the API
    pic_response.close()

    # plt.figure(figsize=(10, 10))
    # img = mpimg.imread('cache/images/test3.jpg')
    # imgplot = plt.imshow(img)
    # plt.show()
    # https://maps.googleapis.com/maps/api/streetview?location=Z%C3%BCrich&size=400x400&key=AIzaSyCf-gm_UHKwm3Rk-Qok1LjIZ-jrVmhGvps
    # 49.482754476094, 8.479343325212051
    # &heading=151.78&pitch=-0.76&
    # staticmap?center = 40.714728, -73.998672
    # "https://maps.googleapis.com/maps/api/elevation/json?locations=349.482754476094%2C8.47934332521205&key=AIzaSyCf-gm_UHKwm3Rk-Qok1LjIZ-jrVmhGvps"
    # https://maps.googleapis.com/maps/api/streetview?location=49.482754476094,8.479343325212051&heading=151.78&pitch=-0.76&&size=640x640&key=AIzaSyCf-gm_UHKwm3Rk-Qok1LjIZ-jrVmhGvps