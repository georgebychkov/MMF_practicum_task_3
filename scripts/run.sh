cd ..
docker run -p 5000:5000 -v $PWD/data:/root/data --rm -i bychkovgk/ensembles
cd scripts