# One entry point, so nothing in the docs depends on which python is on PATH.
#
#   make setup     create .venv and install everything
#   make test      run the suite
#   make run       launch the app against the synthetic room
#   make runtime   fetch the bundled Node + Eufy bridge
#   make model     export the detector
#   make app       build Surface Guard.app
#   make release   build, sign and publish   (VERSION=1.0.0)

PY := .venv/bin/python
PIP := .venv/bin/pip
VERSION ?= 0.0.0

.PHONY: setup test run runtime model app release check clean doctor

$(PY):
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip

setup: $(PY)
	$(PIP) install -q -r requirements-dev.txt
	@echo "Ready. Everything else runs through make, or through $(PY) directly."

doctor: $(PY)
	@$(PY) tools/doctor.py

test: setup
	$(PY) -m pytest -q

run: setup
	PYTHONPATH=src $(PY) -m surfaceguard.app --demo

runtime: setup
	$(PY) packaging/fetch_runtime.py

model: setup
	$(PY) packaging/export_model.py

app: setup
	$(PY) packaging/build_app.py --version $(VERSION) --with-onnx \
		--key $$($(PY) packaging/make_release.py --show-key)

release: setup
	@test "$(VERSION)" != "0.0.0" || (echo "set VERSION, e.g. make release VERSION=1.0.0"; exit 1)
	$(PY) packaging/make_release.py --version $(VERSION) --with-onnx

check: setup
	"dist/Surface Guard.app/Contents/MacOS/Surface Guard" --demo --selftest /tmp/check.png

clean:
	rm -rf build dist
