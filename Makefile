SHELL := /bin/bash

# Folosește direct Python din .venv, deci nu trebuie activat mediul
PY      := .venv/bin/python

# Setări implicite (se pot suprascrie: make train EPOCHS=50 LR=5e-4)
ARCH    ?= cnn
EPOCHS  ?= 30
LR      ?= 1e-3
IMG     ?= 64
NAME    ?= $(ARCH)_e$(EPOCHS)

.PHONY: help data train train-bg logs status stop eval test \
        resnet resnet-eval resnet-test gpu

help:
	@echo "Comenzi disponibile:"
	@echo "  make data           verifică/descarcă EuroSAT și afișează un batch"
	@echo "  make train          antrenare în prim-plan (ARCH, EPOCHS, LR, IMG, NAME)"
	@echo "  make train-bg       antrenare în fundal, log în logs/<NAME>.txt"
	@echo "  make logs           urmărește logul (NAME=...)"
	@echo "  make status         arată dacă rulează o antrenare"
	@echo "  make stop           oprește antrenarea"
	@echo "  make eval           evaluare pe VALIDARE (NAME=..., ARCH=..., IMG=...)"
	@echo "  make test           evaluare pe TEST (cere confirmare)"
	@echo "  make resnet         ResNet18 în fundal (15 epoci, lr 1e-4, 128 px)"
	@echo "  make resnet-eval    evaluează ResNet18 pe validare"
	@echo "  make resnet-test    evaluează ResNet18 pe test (cere confirmare)"
	@echo "  make gpu            monitorizare GPU"

data:
	$(PY) src/dataset.py

train:
	$(PY) src/train.py --arch $(ARCH) --image-size $(IMG) \
		--epochs $(EPOCHS) --lr $(LR) --name $(NAME)

train-bg:
	@mkdir -p logs
	nohup $(PY) src/train.py --arch $(ARCH) --image-size $(IMG) \
		--epochs $(EPOCHS) --lr $(LR) --name $(NAME) \
		> logs/$(NAME).txt 2>&1 &
	@echo "Pornit: $(NAME). Urmărește cu: make logs NAME=$(NAME)"

logs:
	tail -f logs/$(NAME).txt

# Parantezele din [s]rc fac ca pattern-ul să nu se potrivească cu propria comandă
status:
	@pgrep -af "[s]rc/train.py" || echo "Nu rulează nicio antrenare."

stop:
	@pkill -f "[s]rc/train.py" && echo "Oprit." || echo "Nu rula nimic."

eval:
	$(PY) src/evaluate.py --name $(NAME) --arch $(ARCH) \
		--image-size $(IMG) --split val

test:
	@read -p "Rulezi pe setul de TEST pentru $(NAME)? (y/N) " a; [ "$$a" = "y" ]
	$(PY) src/evaluate.py --name $(NAME) --arch $(ARCH) \
		--image-size $(IMG) --split test

resnet:
	$(MAKE) train-bg ARCH=resnet18 EPOCHS=15 LR=1e-4 IMG=128

resnet-eval:
	$(MAKE) eval ARCH=resnet18 EPOCHS=15 IMG=128

resnet-test:
	$(MAKE) test ARCH=resnet18 EPOCHS=15 IMG=128

gpu:
	watch -n 1 nvidia-smi