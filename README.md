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

## How to setup the environment

Follow the instructions in the OpenFaaS documentation to setup the OpenFaaS environment. The instructions can be found at https://docs.openfaas.com/deployment/kubernetes/

Install KinD cluster
`go install sigs.k8s.io/kind@v0.25.0 && kind create cluster`

Install Arkade
`curl -sSL https://get.arkade.dev | sudo -E sh`

Install OpenFaaS
`arkade install openfaas`

Install faas-cli
`arkade get faas-cli`

Deploy the OpenFaaS gateway
`kubectl rollout status -n openfaas deploy/gateway` 
`kubectl port-forward -n openfaas svc/gateway 8080:8080`
`PASSWORD=$(kubectl get secret -n openfaas basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode; echo)`
`echo -n $PASSWORD | faas-cli login --username admin --password-stdin`
`faas-cli store deploy figlet`
`faas-cli list`
`faas-cli store deploy figlet --platform armhf`


## How to deploy the image

You can deploy the image to OpenFaaS by the following command:

`faas-cli deploy --image=leong589/openfaas:latest --name=model`

Alternatively, you can edit the template.yml to change the name of the docker image and deploy it by OpenFaaS.

`faas-cli up -f template.yml`


