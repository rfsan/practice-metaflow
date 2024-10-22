# Metaflow

## Installation

### Local execution + S3 for the datastore

This setup is for running Metaflow locally using S3 as a Datastore. This way you can see the artifacts in the Metaflow UI

1. Rename the file `.metaflow/config.template.json` to `.metaflow/config.json`. Inside the file change the bucket name
2. Be sure you have the AWS credentials at `~/.aws`. They will be copied inside the `metaflow-service` Docker container
3. Clone [netflix/metaflow-service][mf-service] and run it using Docker Compose  

   ```bash
   git clone https://github.com/Netflix/metaflow-service.git
   cd metaflow-service
   AWS_PROFILE=default docker compose -f docker-compose.development.yml up
   ```

4. Clone [netflix/metaflow-ui][mf-ui] and run it using Docker

   ```bash
   git clone https://github.com/Netflix/metaflow-ui.git
   cd metaflow-ui
   docker build --tag metaflow-ui:latest .
   docker run -p 3000:3000 -e METAFLOW_SERVICE=http://localhost:8083/ metaflow-ui:latest
   ```

5. Run `python flows/minimum_flow.py run`


### Configuration file location

- `~/.metaflowconfig/config.json` Global config
- `./.metaflow/config.json` Project overrides

[mf-service]: https://github.com/Netflix/metaflow-service
[mf-ui]: https://github.com/Netflix/metaflow-ui
