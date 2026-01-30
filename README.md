# Vision Risk Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat-square)
![Redis](https://img.shields.io/badge/Redis-Messaging-DC382D?style=flat-square)
![VLM](https://img.shields.io/badge/VLM-SmolVLM--500M-EE4C2C?style=flat-square)

This project implements a multi-service asynchronous pipeline for automated PPE compliance and workplace safety monitoring. The system bridges traditional object detection with advanced visual reasoning to provide contextual risk analysis from live video streams.

## Core Architecture

The system operates through a decoupled Detection + VLM + Redis architecture. A specialized SmartStreamer handles frame ingestion using a sliding-window approach to identify potential risks via Roboflow-integrated inference. Once a detection occurs, the event is queued in Redis, allowing a separate VLM worker to perform asynchronous contextual analysis without bottlenecking the primary vision stream.

## Visual Reasoning with SmolVLM

Unlike standard detection models that only label objects, this system integrates [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) to interpret the scene. This 500M-parameter model provides a high-performance balance between speed and intelligence, allowing the system to describe the specific nature of a risk—such as an individual's proximity to hazardous machinery—rather than simply reporting a missing safety item.

## Deployment and Scalability

The environment is fully containerized using a GPU-aware Docker Compose configuration, ensuring production-ready deployment with seamless access to hardware acceleration. The use of Redis as an event bus ensures that the inference engine and the VLM reasoning layer can scale independently based on workload requirements.

---
*Note: This project is currently in progress. VLM prompt optimization and comprehensive installation guides will be provided in upcoming updates.*