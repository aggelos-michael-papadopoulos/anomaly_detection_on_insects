import cv2
import numpy as np

img = cv2.imread('/home/angepapa/Desktop/papadopoulos corosect/Datasets/counting_dataset/17.png')
print(img.shape)

y, x, channel = img.shape                                   # height, width, channels
y = int(y)
x = int(x)
print(f'shape: {img.shape} \n x = {x} and y = {y}')
# cv2.imshow('original', img)

my = int(y / 2)
mx = int(x / 2)
print(mx, my)

####Scenario1: split image into half ####
scenario1_crop1 = img[:, 0:mx]                                    # right half
scenario1_crop2 = img[:, mx:]                                     # left half


####Scenario2: split image into four ####
Scenario2_crop1 = img[:my, 0:mx]                                    # 1st quarter (left)
Scenario2_crop2 = img[:my, mx:]                                     # 2nd quarter (right)
Scenario2_crop3 = img[my:, 0:mx]                                    # 3rd quarter (left)
Scenario2_crop4 = img[my:, mx:]                                     # 4th quarter (right)

####Scenario3: split into 8 ####
Scenario3_crop1 = img[:my//2, 0:mx]                                 # 1st eighth (left)
Scenario3_crop2 = img[:my//2, mx:]                                  # 2nd eighth (right)
Scenario3_crop3 = img[my-300:my, 0:mx]                              # 3rd eighth (left)
Scenario3_crop4 = img[my-300:my, mx:]                               # 4th eighth (right)
Scenario3_crop5 = img[my:my+300, 0:mx]                              # 5th eighth (left)
Scenario3_crop6 = img[my:my+300, mx:]                               # 6th eighth (right)
Scenario3_crop7 = img[my+300:, 0:mx]                                # 7rd eighth (left)
Scenario3_crop8 = img[my+300:, mx:]                                 # 8th eighth (right)

####Scenario4: split into 16 ####
Scenario4_crop1 = img[:my//4, 0:mx]                           # 1st sixteenth (full left)
Scenario4_crop2 = img[:my//4, mx:]                            # 2nd sixteenth (full right)
Scenario4_crop3 = img[my//2//2:my-300, 0:mx]
Scenario4_crop4 = img[my//2//2:my-300, mx:]
Scenario4_crop5 = img[my-150:my, 0:mx]                              # 3rd sixteenth (left) ?
Scenario4_crop6 = img[my-150:my, mx:]                               # 4th sixteenth (right) ?
Scenario4_crop7 = img[my:my+150, 0:mx]                              # 5th sixteenth (left)
Scenario4_crop8 = img[my:my+150, mx:]                               # 6th sixteenth (right)
Scenario4_crop9 = img[my+450:, 0:mx]                                # 7rd sixteenth (left)
Scenario4_crop10 = img[my+450:, mx:]                                 # 8th sixteenth (right)
Scenario4_crop11 = img[my-450:my//2, 0:mx]                                 # 1st sixteenth (left)
Scenario4_crop12 = img[my-450:my//2, mx:]                                 # 2nd sixteenth (right)
Scenario4_crop13 = img[my-300:my-150, 0:mx]                             # 3rd sixteenth (left)
Scenario4_crop14 = img[my-300:my-150, mx:]                              # 4th sixteenth (right)
Scenario4_crop15 = img[my+150:my+300, 0:mx]                             # 5th sixteenth (left)
Scenario4_crop16 = img[my+150:my+300, mx:]                              # 6th sixteenth (right)


cv2.imwrite('scenario1_crop1.png', scenario1_crop1)
cv2.imwrite('scenario1_crop2.png', scenario1_crop2)

cv2.imwrite('Scenario2_crop1.png', Scenario2_crop1)
cv2.imwrite('Scenario2_crop2.png', Scenario2_crop2)
cv2.imwrite('Scenario2_crop3.png', Scenario2_crop3)
cv2.imwrite('Scenario2_crop4.png', Scenario2_crop4)

cv2.imwrite('Scenario3_crop1.png', Scenario3_crop1)
cv2.imwrite('Scenario3_crop2.png', Scenario3_crop2)
cv2.imwrite('Scenario3_crop3.png', Scenario3_crop3)
cv2.imwrite('Scenario3_crop4.png', Scenario3_crop4)
cv2.imwrite('Scenario3_crop5.png', Scenario3_crop5)
cv2.imwrite('Scenario3_crop6.png', Scenario3_crop6)
cv2.imwrite('Scenario3_crop7.png', Scenario3_crop7)
cv2.imwrite('Scenario3_crop8.png', Scenario3_crop8)

cv2.imwrite('Scenario4_crop1.png', Scenario4_crop1)
cv2.imwrite('Scenario4_crop2.png', Scenario4_crop2)
cv2.imwrite('Scenario4_crop3.png', Scenario4_crop3)
cv2.imwrite('Scenario4_crop4.png', Scenario4_crop4)
cv2.imwrite('Scenario4_crop5.png', Scenario4_crop5)
cv2.imwrite('Scenario4_crop6.png', Scenario4_crop6)
cv2.imwrite('Scenario4_crop7.png', Scenario4_crop7)
cv2.imwrite('Scenario4_crop8.png', Scenario4_crop8)
cv2.imwrite('Scenario4_crop9.png', Scenario4_crop9)
cv2.imwrite('Scenario4_crop10.png', Scenario4_crop10)
cv2.imwrite('Scenario4_crop11.png', Scenario4_crop11)
cv2.imwrite('Scenario4_crop12.png', Scenario4_crop12)
cv2.imwrite('Scenario4_crop13.png', Scenario4_crop13)
cv2.imwrite('Scenario4_crop14.png', Scenario4_crop14)
cv2.imwrite('Scenario4_crop15.png', Scenario4_crop15)
cv2.imwrite('Scenario4_crop16.png', Scenario4_crop16)

cr = cv2.imread('scenario1_crop1.png')
cr2 = cv2.imread('scenario1_crop2.png')

cr3 = cv2.imread('Scenario2_crop1.png')
cr4 = cv2.imread('Scenario2_crop2.png')
cr5 = cv2.imread('Scenario2_crop3.png')
cr6 = cv2.imread('Scenario2_crop4.png')

cr7 = cv2.imread('Scenario3_crop1.png')
cr8 = cv2.imread('Scenario3_crop2.png')
cr9 = cv2.imread('Scenario3_crop3.png')
cr10 = cv2.imread('Scenario3_crop4.png')
cr11 = cv2.imread('Scenario3_crop5.png')
cr12 = cv2.imread('Scenario3_crop6.png')
cr13 = cv2.imread('Scenario3_crop7.png')
cr14 = cv2.imread('Scenario3_crop8.png')

cr15 = cv2.imread('Scenario4_crop1.png')
cr16 = cv2.imread('Scenario4_crop2.png')
cr17 = cv2.imread('Scenario4_crop3.png')
cr18 = cv2.imread('Scenario4_crop4.png')
cr19 = cv2.imread('Scenario4_crop5.png')
cr20 = cv2.imread('Scenario4_crop6.png')
cr21 = cv2.imread('Scenario4_crop7.png')
cr22 = cv2.imread('Scenario4_crop8.png')
cr23 = cv2.imread('Scenario4_crop9.png')
cr24 = cv2.imread('Scenario4_crop10.png')
cr25 = cv2.imread('Scenario4_crop11.png')
cr26 = cv2.imread('Scenario4_crop12.png')
cr27 = cv2.imread('Scenario4_crop13.png')
cr28 = cv2.imread('Scenario4_crop14.png')
cr29 = cv2.imread('Scenario4_crop15.png')
cr30 = cv2.imread('Scenario4_crop16.png')

print(cr.shape)
print(cr2.shape, '\n')

print(cr3.shape)
print(cr4.shape)
print(cr5.shape)
print(cr6.shape, '\n')

print(cr7.shape)
print(cr8.shape)
print(cr9.shape)
print(cr10.shape)
print(cr11.shape)
print(cr12.shape)
print(cr13.shape)
print(cr14.shape, '\n')

print(cr15.shape)
print(cr16.shape)
print(cr17.shape)
print(cr18.shape)
print(cr19.shape)
print(cr20.shape)
print(cr21.shape)
print(cr22.shape)
print(cr23.shape)
print(cr24.shape)
print(cr25.shape)
print(cr26.shape)
print(cr27.shape)
print(cr28.shape)
print(cr29.shape)
print(cr30.shape, '\n')
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.waitKey(0)
# cv2.destroyAllWindows()