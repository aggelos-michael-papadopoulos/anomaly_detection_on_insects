from model import AnomalyAE
import torch
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt
import cv2

def anomaly_detection(image_path, model, loading_weights):
    model.load_state_dict(torch.load(loading_weights))
    model.eval()
    model = model.to('cuda')

    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))
    transform = Compose([ToTensor()])
    img = transform(img)
    img = img.to('cuda')
    img = img.unsqueeze(0)
    y = model(img)
    residual = torch.abs(img[0][0] - y[0][0])

    plt.figure(figsize=(15, 10))
    plt.subplot(121);
    plt.imshow(img.detach().cpu().numpy()[0][0]);
    plt.title('Image')
    plt.axis('off');
    plt.subplot(122);
    plt.imshow(residual.detach().cpu().numpy() > 0.085);  # 0.015
    plt.title('Residual Thresholded')
    plt.axis('off');
    plt.savefig('sample_detection.png', bbox_inches='tight')



an = anomaly_detection('EntocycleEntocycle_13DOL_458frame.png', AnomalyAE(), 'ALL_DATA_NEW: tensorboard_logs_10052022_19-58/models/best_model_4_loss=-0.0003.pt')


