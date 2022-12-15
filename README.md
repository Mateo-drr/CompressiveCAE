# CompressiveCAE

This CAE is an improved version I've been working on for the last year as a side project from my bachelor's final year project A2VO, which you can also find the structure in the separate folder with the complete report.

- Main differences: activation functions, jump connections, dataset preprocessing, latent space compression and the use of pixelunshuffle along side the previously used pixelshuffle

# Training information
  - Relevant graphs and results can be found in the RESULTS folder
  - The resConvfmod structure presents a lower loss but in the test images showed similar results to the resConv structure. It's main difference appears to be in PSNR
  - Due to limited hardware availability the structures were trained until a maximum of 185 epochs with a cropped stanford cars dataset which is explained in the original paper

# How to use
  - The script includes the final two versions tested, resConv and resConvfmod. The only difference between them is a jump connection. In the script you can change line 418 to the name of the one you would like to use

# Future plans
  - I'll be working on a GUI for the project to make it a viable compressing tool
  - Plan on adding an extra compression option that would use blur on the objects in the background adding a boca effect to the pictures and thus hopefully reducing the amount of high frequency elements in the pictures, reducing the space needed for compression.
  
# Current Results
  - The results from the different structures can be found here: https://docs.google.com/spreadsheets/d/1O-tJ4gmN9WLuyeUp-tY2MAou6w0m0acFocCmNF2HcbM/edit?usp=sharing
  
