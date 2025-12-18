FROM python:3.11.9-slim-bookworm

WORKDIR /mnt/locust
RUN pip install locust

EXPOSE 8089

CMD sh -c 'locust --host="$TARGET_HOST" --web-host=0.0.0.0'
