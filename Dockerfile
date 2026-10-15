FROM python:3.12-slim
WORKDIR /app

# Same corporate TLS-inspecting gateway note as the React version's
# server/Dockerfile — needed for outbound HTTPS (Groq/DuckDuckGo/Google) to
# work from inside this container on this network.
COPY cloudflare-gateway-ca.pem /usr/local/share/ca-certificates/cloudflare-gateway-ca.pem
RUN cat /usr/local/share/ca-certificates/cloudflare-gateway-ca.pem >> $(python -c "import certifi; print(certifi.where())") 2>/dev/null || true
ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/cloudflare-gateway-ca.pem
ENV SSL_CERT_FILE=/usr/local/share/ca-certificates/cloudflare-gateway-ca.pem

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && cat /usr/local/share/ca-certificates/cloudflare-gateway-ca.pem >> $(python -c "import certifi; print(certifi.where())")

COPY . .
RUN mkdir -p /app/data

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
