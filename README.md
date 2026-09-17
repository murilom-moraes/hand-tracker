# Hand Tracker

Rastreamento de mãos em tempo real utilizando Python, OpenCV e MediaPipe.

O projeto utiliza a webcam para detectar uma mão, identificar suas 21 articulações e visualizar suas conexões em tempo real.

## Funcionalidades

- Detecção e rastreamento de mãos em tempo real.
- Visualização dos 21 pontos da mão e suas conexões.
- Identificação da mão esquerda e direita.
- Posicionamento dinâmico do texto, acompanhando um dedo levantado.
- Exibição de FPS em tempo real.

## Tecnologias

- Python 3.14
- OpenCV
- MediaPipe

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/murilom-moraes/hand-tracker.git
cd hand-tracker
```

### 2. Instale as dependências

```bash
python -m pip install -r requirements.txt
```

Recomenda-se utilizar um ambiente virtual Python.

### 3. Baixe o modelo

Baixe o [modelo Hand Landmarker do MediaPipe](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).

Coloque o arquivo `hand_landmarker.task` na pasta principal do projeto, junto ao `main.py`.

## Execução

Execute o programa:

```bash
python main.py
```

Pressione `Q` para encerrar a aplicação.

## Requisitos

- Windows
- Python 3.14.5 (versão testada)
- Webcam

## Licença

Este projeto está disponível sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais informações.