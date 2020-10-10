import argparse
import glob
import os
import pdb
import pandas as pd
import csv

import numpy                  as np
import PIL.Image              as Image
import torch
from   torch.utils.data       import DataLoader, TensorDataset
import torchvision.transforms as     transforms
from   tqdm                   import tqdm

torch.nn.Module.dump_patches = True

parser = argparse.ArgumentParser(description='scripts to apply saved models to test data')
parser.add_argument('trained_model', help='trained model')
parser.add_argument('test_dir',      help='directory contains test images.') 
parser.add_argument('--patchsize',  default=224,   type=int,   help='test patch size')
parser.add_argument('--batchsize',  default=32,    type=int,   help='test batch size')
args = parser.parse_args()
print(args)

device = torch.device('cuda')


print('===> Loading data')
# get image path
test_image_pathes = glob.glob(os.path.join(args.test_dir, '*.png'))
test_image_pathes.sort()
assert len(test_image_pathes) != 0, 'no data in test_image_patches'

# convert image into tensor and normalize
transform = transforms.Compose([
                transforms.CenterCrop(args.patchsize),
                transforms.ToTensor()
            ])
image_tensors = [transform(Image.open(path)).unsqueeze(0) for path in test_image_pathes]
tmp           = torch.Tensor(len(image_tensors), 3, args.patchsize, args.patchsize)
image_tensors = torch.cat(image_tensors, out=tmp)

# make dataloader(optional)
test_dst        = TensorDataset(image_tensors)
test_dataloader = DataLoader(test_dst,
                             batch_size=args.batchsize,
                             shuffle=False,
                             num_workers=4)


print('===> Loading model')
model = torch.load(args.trained_model)
model.to(device)
model.eval()



#pdb.set_trace()

print('===> Predicting...')
#outputs = pd.DataFrame(columns=[ 'outputs'])
predictions = []
log=[]
for batch in test_dataloader:
    with torch.no_grad():
        input  = batch[0]
        input  = input.to(device)
        output = model(input).cpu().numpy()
        #n_output = torch.nn.functional.softmax(output, dim=0)
        log.extend(output)
        prediction = np.argmax(output, axis=1)
        predictions.extend(prediction)
        #outputs = pd.DataFrame([(output)], columns=['outputs'])



print('===> Output as csv.')
test_image_names = [path.split('/')[-1] for path in test_image_pathes]

if not os.path.exists('results'):
    os.mkdir('results')

model_name  = args.trained_model.split('/')[-1].split('.')[0]
output_path = os.path.join('probs', model_name + '.csv')

z = np.asarray(log)
np.savetxt("num_log.csv", z, delimiter=",")

#with open('outputs3.csv', 'w') as f:
#    wr=csv.writer(f, quoting=csv.QUOTE_ALL)
#    wr.writerow(log)

#with open(output_path, 'w') as f:
#    for name, pred in zip(test_image_names, output):
#        f.write('{},{}\n'.format(name, pred)
