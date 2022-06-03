import os




datapath = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/counting_dataset/'
splitted_datapath = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/counting_dataset/splitted/'


images = []
for img in os.listdir(datapath):
    if '.png' in img:
        img_id = int(img.split('.')[-2])
        images.append(img_id)

images_in_order = []
images = sorted(images)

for i in images:
    final = str(i) + '.png'
    # final_path = os.path.join(datapath, final)
    images_in_order.append(final)

# print(images_in_order)


# labels = [23, 54, 65, 87, 52]
img_limit = 100
img_packs = []
for i in range(31):
    pack = images_in_order[img_limit*i:img_limit*(i+1)]                                 #[0:99] -> [100:199]
    img_packs.append(pack)

print(f'We have {len(img_packs)} packs.\n'
      f'Each pack contains {img_limit} images. \n'
      f'So the whole packs are a {len(img_packs)} length list where each element of this list is a list that contains '
      f'{img_limit} images: {img_packs}')







