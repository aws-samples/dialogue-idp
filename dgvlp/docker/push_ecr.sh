# Define docker image name and container's Amazon Reource Name on ECR
container_name="tgi1.03"
region=`aws configure get region`
account=`aws sts get-caller-identity --query "Account" --output text`
full_name="${account}.dkr.ecr.${region}.amazonaws.com/${container_name}:latest"

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region}|docker login --username AWS \
    --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

# Build the TGI docker image locally
docker build . -f Dockerfile -t ${container_name}
docker tag ${container_name} ${full_name}
docker push ${full_name}
