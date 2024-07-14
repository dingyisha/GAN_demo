FROM mcr.microsoft.com/devcontainers/python:3.8

# 安装必要的软件包
RUN apt-get update && apt-get install -y wget gnupg2 unzip
RUN pip install pandas requests beautifulsoup4 selenium

# 安装 Chrome 浏览器
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get update && apt-get install -y google-chrome-stable

# 下载并安装 ChromeDriver，适用于 126.0.6478.114
RUN wget -O /tmp/chromedriver.zip https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip && \
    unzip /tmp/chromedriver.zip -d /usr/local/bin/ && \
    rm /tmp/chromedriver.zip

# 设置默认工作目录
WORKDIR /workspace




