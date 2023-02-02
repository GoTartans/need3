import sys
import kasa
import asyncio
import numpy as np
from interpolate import interpolate_color

class Bulb:
    def __init__(self):
        self.hues = {'anger' : (0, 67, 15),
            'disgust' : (137, 30, 8),
            'fear': (313, 80, 31),
            'joy' : (56, 31, 8),
            'neutral' : (0, 0, 5),
            'sad' : (208, 88, 15),
            'surprise' : (353, 100, 15)
        }
        self.bulb = kasa.SmartBulb("172.26.174.241")

    async def change_color(self, logit):
        threshold = 1/7

        argsort = np.argsort(logit)[::-1]
        primary_idx, secondary_idx = argsort[0], argsort[1]

        await asyncio.sleep(0.1)
        await self.bulb.update()
        if logit[secondary_idx] < threshold:
            try:
                # print(list(hues.keys())[primary_idx])
                await self.bulb.set_hsv(*list(self.hues.values())[primary_idx])
            except:
                await self.bulb.set_hsv(*self.hues['neutral'])
        else:
            primary_hue = list(self.hues.values())[primary_idx]
           
            try:
                await self.bulb.set_hsv(*primary_hue)
            except Exception as e:
                print(e)
                await self.bulb.set_hsv(*self.hues['neutral'])

if __name__ == '__main__':
    bulb = Bulb() 
    if len(sys.argv) > 2:
        emotion = list(map(float, sys.argv[1:8]))
        asyncio.run(bulb.change_color(emotion))

