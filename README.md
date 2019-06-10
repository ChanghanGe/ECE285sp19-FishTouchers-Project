# ECE285sp19-FishTouchers-Project

![](frog_scream.gif)

## Description
This is project Style Transfer developed by team FishTouchers composed of
<ul>
  <li><strong>Changhan Ge</strong>   A53264280</li>
  <li><strong>Yifan Huang</strong>   A53272363</li>
  <li><strong>Song Wang</strong>     A53275641</li>
  <li><strong>Jingqi Huang</strong>  A53281935</li>
</ul>

## Requirements
Install packages as follow:

## Code organzation
### Codes
demo.ipynb                        --  Run a demo on a trained transformation network (You can change the 'style' or 'content' strings in the code to vary the content/style images) <br />
Part_A_NeuralStyleTransfer.ipynb  --  Gytes' style transfer network training program.<br />
Part_B_RealTimeTransfer.ipynb     --  Feifei's real-time style transfer network training program.<br />
trans_net.py                      --  Define the transformation network architecture. <br />
vgg.py                        --  Define the loss network architecture for the Part_A_NeuralStyleTransfer.ipynb. <br />

### Files
<strong>Demo-frog-3.mp4</strong>  --  <strong>Style transferred video mentioned in report Sec.4.5.</strong><br />
Models                            --  A collection of trained transformation networks.<br />
images                            --  A collection of style/content images. The style images correspond to the trained networks in Models. You can choose different content images in demo.ipynb to check the transferred result.<br />
