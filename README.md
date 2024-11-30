[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jzfQvm5J)
# Project -- Deployment and optimization of machine learning models in a serverless architecture

The repository contains the code for the project "Deployment and optimization of machine learning models in a serverless architecture" for the course "Cloud Computing and Big Data" at the Hong Kong University of Science and Technology.

The report is in the root directory of the repository and is named `report.pdf`.
## Team members
Name: Ngai Lai Yeung
Email: lyngai@connect.ust.hk

Name: Zhang Zi Di
Email: zzhangcy@connect.ust.hk

## Project Description
The repository is a dockerfile that can be build by OpenFaaS to deploy a machine learning model.The image is also available on my dockerhub leong589/testing:latest

You can deploy the image to OpenFaaS by the following command:

`faas-cli deploy --image=leong589/testing:latest --name=model`

Alternatively, you can edit the template.yml to change the name of the docker image and deploy it by OpenFaaS.

`faas-cli up -f template.yml`


