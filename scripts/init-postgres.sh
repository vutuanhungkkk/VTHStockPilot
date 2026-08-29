#!/bin/bash
# Creates multiple PostgreSQL databases for stockpilot + mlflow
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE stockpilot TO $POSTGRES_USER;
EOSQL
