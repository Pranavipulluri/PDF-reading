FROM python:3.11-slim 
RUN pip install PyMuPDF==1.23.8 
WORKDIR /app 
COPY main.py ./ 
COPY run.sh ./ 
RUN chmod +x run.sh && mkdir -p input output 
CMD ["./run.sh"] 
