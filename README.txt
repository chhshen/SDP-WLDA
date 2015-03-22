
This is the Matlab code associated with the publication:
"Worst-Case Linear Discriminant Analysis as Scalable Semidefinite Feasibility Problems." 
by Hui Li, Chunhua Shen, Anton van den Hengel and Qinfeng Shi. IEEE T. Image Processing 2015


Email contact: hui.li02@adelaide.edu.au

This document walks through how to calculate the transformation matrix for worst-case linear discriminant analysis based on an efficient semidefinite programming approach. The core parts are:

I. SD_WLDA.m: solve the relaxed WLDA problem by bisection search.

II. SDPAlg.m: Solve the SDP feasibility problem in each step of bisection search.

The method on how to use those functions are shown in the files respectively.
======================================================================
Prerequisites
======================================================================
- Install L-BFGS-B toolbox.
  L-BFGS-B mex interface is needed to solve the SDP problem. There are two wrappers here (lbfgsb_wrapper.mexmaci64 for Mac and lbfgsb_wrapper.mexa64 for Linux), which are provided by Stephen Becker. Users need to compile a suitable lbfgsb mex interface for their operation system.
  
======================================================================
QUICK DEMO
======================================================================

This demo shows how to get the transformation matrix based on the proposed SD-WLDA algorithm. The demo uses Iris dataset from UCI. 

>> Demo


if you use this code, please cite our paper:

 @article{SDP2015Li,
   author    = "H. Li and  C. Shen and  A. {van den Hengel} and  Q. Shi",
   title     = "Worst-case linear discriminant analysis as scalable semidefinite feasibility problems",
   journal   = "IEEE Transactions on Image Processing",
   year      = "2015",
 }
 
