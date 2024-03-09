#!/bin/zsh

wget -P .. https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz
tar -xf ../nuimages-v1.0-mini.tgz

wget -P .. https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0/nuimages-v1.0-all-samples.tgz
tar -xf ../nuimages-v1.0-all-metadata.tgz

wget -P .. https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0/nuimages-v1.0-all-metadata.tgz
tar -xf ../nuimages-v1.0-all-sample.tgz



