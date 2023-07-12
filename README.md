# Dialogue-IDP

<p>Dialogue Guided Intelligent Document Processing (DGIDP) is an innovative approach to extracting and processing information from documents by leveraging natural language understanding and conversational AI. This technique allows users to interact with the IDP system using human-like conversations, asking questions, and receiving relevant information in real-time. The system is designed to understand context, process unstructured data, and respond to user queries effectively and efficiently.</p> <p>While the text or voice chat accepts all major languages, the document upload feature only accepts files in English, German, French, Spanish, Italian, and Portuguese. The demo supports <u>multilingual text and voice</u> input, as well as <u>multi-page</u> documents in PDF, PNG, JPG, or TIFF format.</p>

<p>To use SageMaker Jumpstart foundation model for text generation, use the notebook to deploy an endpoint and test.</p>

<p>To use third-party APIs such as OpenAI APIs and SERP APIs, you might risk sharing your private information with third-party API providers. Be caucious of your senstive information.</p>
<p>This guidance is for informational purposes only.  You should still perform your own independent assessment, and take measures to ensure that you comply with your own specific quality control practices and standards, and the local rules, laws, regulations, licenses and terms of use that apply to you, your content, and the third-party generative AI service referenced in this guidance.  AWS has no control or authority over the third-party generative AI service referenced in this guidance, and does not make any representations or warranties that the third-party generative AI service is secure, virus-free, operational, or compatible with your production environment and standards. AWS does not make any representations, warranties or guarantees that any information in this guidance will result in a particular outcome or result.</p>

## Config environment variables
<p>This code has been tested on EC2 server with al2023-ami-2023.0.20230503.0-kernel-6.1-x86_64 AMI type. You will need to configure you security group to allow inbound traffic to port 7862.</p>

<p>In your server, locate env_var_conf file, save your tokens in the file, and excute the following command.</p>

```bat
source ./env_var_conf
```

## Set up server
```Python
python demo.py
```
Go to https://your-server-ip:7862 and choose either dgidp or babyagi to test dialogue IDP. You will need to first upload a document, click Transcribe button, and type your question to Textbox and then click Text Chat button.


