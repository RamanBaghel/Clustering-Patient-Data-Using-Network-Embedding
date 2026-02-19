#!/bin/bash
if [ -z "$1" ]; then
	echo "No name for database supplied"
elif [ -z "$2" ]; then
	echo "No number of patients to simulate supplied"
else
	psql -c "DROP DATABASE IF EXISTS $1"
	psql -c "CREATE DATABASE $1"

	psql $1 < synthea.sql
	java -jar synthea-with-dependencies.jar -c synthea.properties -p $2
	synthea_tables=(patients organizations providers payers encounters allergies careplans claims claims_transactions conditions devices imaging_studies immunizations medications observations payer_transitions procedures supplies)
	for name in "${synthea_tables[@]}"
	do
		pv output/csv/$name.csv | psql $1 -c "COPY synthea_"$name" FROM stdin DELIMITER ',' CSV header"
	done

	python3 clean.py $1
fi
