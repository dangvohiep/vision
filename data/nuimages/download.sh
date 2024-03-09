#!/bin/zsh

wget -P .. https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz
tar -xf ../nuimages-v1.0-mini.tgz

wget -P .. https://www.nuscenes.org/data/nuimages-v1.0-train.tgz
tar -xf ../nuimages-v1.0-train.tgz

wget -P .. https://www.nuscenes.org/data/nuimages-v1.0-val.tgz
tar -xf ../nuimages-v1.0-val.tgz

wget -P .. https://www.nuscenes.org/data/nuimages-v1.0-tessdft.tgz
tar -xf ../nuimages-v1.0-test.tgz

wget -P .. https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0/nuimages-v1.0-all-samples.tgz
wget -P .. https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0/nuimages-v1.0-all-metadata.tgz



