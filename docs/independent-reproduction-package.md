# Zamrożony pakiet dla niezależnej reprodukcji

Kontrakt pakietu definiuje `configs/reproduction/input-package.yaml`. Plik
`release-metadata.json` jest generowany podczas wydania i musi zawierać podpisany tag,
commit, SHA-256 archiwum, czas UTC, platformę wheelhouse oraz tożsamość osoby wydającej.
Sam kontrakt nie udaje gotowego wydania: pakiet jest ważny dopiero po obecności tych danych
i pomyślnej weryfikacji. Zespół reprodukcyjny otrzymuje jedno archiwum read-only oraz
oddzielny podpis, bez wyników testowych autorów.

## Zawartość i budowa

1. Z czystego checkoutu podpisanego tagu utworzyć `git archive`, wheel projektu oraz
   wheelhouse wszystkich przechodnich zależności dla wskazanej platformy. Lock zawiera
   dokładne wersje, nazwy plików, licencje i SHA-256; instalacja używa `--no-index` oraz
   `--require-hashes`.
2. Dołączyć manifesty wersji/checksum danych (nie dane objęte licencją), osobne artefakty
   o udokumentowanym zbiorze i splitach treningowych, konfiguracje, protokoły, skrypty
   metryk oraz pusty szablon raportu zgodności. Każdy plik obejmuje `checksums.sha256`.
3. Budować w jednorazowym kontenerze z wyłączoną siecią i pustymi `HOME`, `XDG_CACHE_HOME`
   oraz `PIP_CACHE_DIR`; nie kopiować `.git`, `outputs`, środowisk, cache'y ani logów autora.
   Uruchomienie, które odczytuje plik spoza rozpakowanego pakietu/danych wejściowych,
   jest nieważne.
4. Wygenerować `checksums.sha256` w stabilnym porządku dla wszystkich plików poza samym
   spisem, podpisać archiwum i uruchomić poniższe polecenie. Nie wolno ręcznie poprawiać
   gotowego archiwum.

   ```bash
   s4dtam-bench verify-package <root> configs/reproduction/input-package.yaml
   ```

## Odbiór

Niezależny zespół weryfikuje podpis i `sha256sum -c`, instaluje offline, uruchamia testy,
`s4dtam-bench validate-ablation configs/experiments/ablation.yaml`, a potem analizę bez
zmiany plików. System, sterowniki, CPU/GPU/RAM, obraz kontenera i każde odstępstwo trafiają
do `hardware.json` lub raportu. Brakująca zależność nie może być „doinstalowana” bez nowej,
wersjonowanej edycji pakietu.

Replikator zapisuje wynik każdego kroku, kod wyjścia i czas UTC. Oczekiwane artefakty to
wierszowy raport zgodności, tabela H1–H7, log kompletności, raport różnic względem publikacji
i manifest wygenerowanych plików. Tolerancje numeryczne muszą pochodzić z metadanych
wydania; brak tolerancji oznacza wymaganie zgodności bitowej dla danego artefaktu.
