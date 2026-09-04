# Raport code review

**Data przeglądu:** 2026-09-04  
**Zakres:** kod Python w `src/`, testy, konfiguracja pakietu i workflow CI  
**Charakter raportu:** lista wymaganych poprawek; przegląd nie obejmował walidacji
wyników naukowych na rzeczywistych zbiorach danych ani audytu zależności zewnętrznych.

## Podsumowanie

Testy jednostkowe przechodzą przy udostępnieniu katalogu `src` na `PYTHONPATH`,
a Ruff nie zgłasza błędów.
Znaleziono jednak **6 wymaganych poprawek**: 3 o priorytecie wysokim i 3 o
priorytecie średnim. Najpoważniejsze problemy mogą prowadzić do opublikowania
metryk policzonych dla niezsynchronizowanych próbek albo dla niezgodnych kształtów
tablic. Przed wykorzystaniem benchmarku do wyników publikacyjnych należy zamknąć
co najmniej pozycje CR-01--CR-04.

## Wymagane poprawki

### CR-01 — Wysoki — ewaluator ignoruje oś czasu wyniku

**Miejsce:** `src/s4dtam_benchmark/evaluation/runner.py:21-24`,
`src/s4dtam_benchmark/contracts.py:133-143`.

**Problem:** `AlgorithmResult` sprawdza jedynie, czy jego znaczniki czasu są
skończonym wektorem. Nie wymaga ich ścisłej monotoniczności, a `evaluate_result`
nie porównuje ich ze znacznikami czasu sekwencji. Pozycje są odejmowane wprost
według indeksu. Artefakt o tej samej liczbie pozycji, ale przesunięty w czasie,
może więc uzyskać idealne ATE. Przy różnej liczbie próbek wykonanie kończy się
mało diagnostycznym błędem broadcastingu zamiast kontrolowanym odrzuceniem.

**Wpływ:** ciche wygenerowanie niepoprawnych wyników porównania trajektorii,
niepewności i ryzyka selektywnego; naruszenie podstawowego kontraktu benchmarku.

**Wymagana poprawka:** przed obliczeniem metryk wymagać zgodności osi czasu
(liczby próbek i wartości z jawnie ustaloną tolerancją) albo wdrożyć i
udokumentować deterministyczną interpolację/asocjację czasową. Dodatkowo wymagać
niepustych, ściśle rosnących znaczników w `AlgorithmResult`.

**Test regresyjny:** wynik z `timestamps = sequence.timestamps + 0.5` ma zostać
odrzucony; osobny test powinien pokryć różną liczbę próbek i niemonotoniczną oś.

### CR-02 — Wysoki — niezgodne kształty predykcji mogą być cicho broadcastowane

**Miejsce:** `src/s4dtam_benchmark/contracts.py:113-130`,
`src/s4dtam_benchmark/evaluation/runner.py:73-88`,
`src/s4dtam_benchmark/evaluation/semantic.py:6-25` oraz
`src/s4dtam_benchmark/evaluation/forecast.py:21-43`.

**Problem:** kontrakt normalizuje i waliduje pozycje, kowariancje oraz wyniki OOD,
ale nie waliduje m.in. `semantic_pred`, `occupancy_pred`, `flow_pred`, masek,
`risk_pred`, `estimated_quaternions` i `latency_ms`. Ewaluatory wykonują działania
NumPy bez sprawdzenia identycznych kształtów. Przykładowo target semantyczny o
kształcie `(N,)` i predykcja `(N, 1)` tworzą przez broadcasting macierz `(N, N)`
i zwracają wiarygodnie wyglądającą, lecz błędną dokładność zamiast wyjątku.

**Wpływ:** ciche zafałszowanie metryk semantycznych i prognostycznych; w innych
wariantach awaria pojedynczego uruchomienia dopiero w ewaluatorze.

**Wymagana poprawka:** rozszerzyć `AlgorithmResult.__post_init__` oraz granicę
`evaluate_result` o walidację typu, skończoności, zakresów prawdopodobieństwa,
wiodącego wymiaru próbek i dokładnej zgodności kształtów target/predykcja/maska.
Nie polegać na regułach broadcastingu NumPy w metrykach.

**Test regresyjny:** dla każdego opcjonalnego wyjścia dodać przypadki poprawne
oraz przypadki z `(N,)` kontra `(N, 1)`, inną długością, `NaN` i (dla
prawdopodobieństw) wartościami poza `[0, 1]`.

### CR-03 — Wysoki — ścieżki względne konfiguracji zależą od katalogu uruchomienia

**Miejsce:** `src/s4dtam_benchmark/config.py:17-20`,
`src/s4dtam_benchmark/experiment.py:31-55`, `58-127` i `131-139`.

**Problem:** istnieje pomocnicza funkcja `resolve_from_config`, lecz konstrukcja
datasetów, mapy referencyjnej i artefaktów zewnętrznych jej nie używa. Także
`output_dir` jest rozwiązywany względem `Path.cwd()`. Ten sam plik YAML uruchomiony
z innego katalogu odczyta inne dane/artefakty lub zapisze wyniki w innym miejscu.

**Wpływ:** brak przenośności i powtarzalności konfiguracji; możliwe przypadkowe
użycie niewłaściwego zbioru danych lub artefaktu bazowego.

**Wymagana poprawka:** jednoznacznie ustalić semantykę ścieżek (zalecane: względem
katalogu pliku konfiguracyjnego), zastosować ją do wszystkich pól ścieżkowych i
zapisać w manifeście ścieżki rozwiązane. Jeżeli `output_dir` ma celowo pozostać
względem CWD, należy to oddzielnie udokumentować i przetestować.

**Test regresyjny:** uruchomić ten sam tymczasowy YAML z dwóch różnych CWD i
potwierdzić identyczny wybór wejść oraz oczekiwane miejsce wyjścia.

### CR-04 — Średni — eksporter MARSIM akceptuje plik, którego adapter odrzuca

**Miejsce:** `src/s4dtam_benchmark/datasets/marsim.py:25-42` oraz
`src/s4dtam_benchmark/contracts.py:49-57`.

**Problem:** eksporter odrzuca tylko ujemną różnicę czasu (`diff < 0`), więc
akceptuje duplikaty znaczników i zapisuje NPZ. `SequenceData` wymaga natomiast
ścisłego wzrostu (`diff <= 0`), przez co świeżo wyeksportowany artefakt nie daje
się następnie wczytać. Losowy tie-breaker zmienia wyłącznie kolejność rekordów o
tym samym czasie, nie usuwa sprzeczności.

**Wpływ:** opóźniona awaria pipeline'u i powstanie nieużywalnych artefaktów.

**Wymagana poprawka:** odrzucać duplikaty już w eksporcie czytelnym błędem albo
zdefiniować deterministyczną agregację/deduplikację przed zapisem.

**Test regresyjny:** eksport dwóch próbek z tym samym timestampem musi zakończyć
się kontrolowanym błędem; każdy pomyślny eksport powinien dać się natychmiast
odczytać przez `MARSIMDataset`.

### CR-05 — Średni — błędy kalibracji omijają mechanizm ewidencji awarii

**Miejsce:** `src/s4dtam_benchmark/experiment.py:145-165`.

**Problem:** właściwe uruchomienia algorytmów są izolowane blokiem `try`, a błędy
są zapisywane w `failures.json`. Wywołania `calibrate` odbywają się wcześniej,
poza tym mechanizmem. Awaria kalibracji jednego algorytmu przerywa cały eksperyment
i nie tworzy raportu awarii, nawet gdy pozostałe algorytmy mogłyby zostać ocenione.

**Wpływ:** niespójna odporność runnera i utrata śladu audytowego dla typowego
punktu awarii.

**Wymagana poprawka:** jawnie wybrać politykę fail-fast albo izolację algorytmów.
W drugim wariancie zapisywać awarię kalibracji, oznaczać algorytm jako niedostępny
i kontynuować pozostałe uruchomienia. W wariancie fail-fast zapisać atomowy raport
diagnostyczny przed ponownym zgłoszeniem wyjątku.

**Test regresyjny:** dwa algorytmy, z których pierwszy zgłasza wyjątek w
`calibrate`; test ma zweryfikować udokumentowaną politykę i obecność diagnostyki.

### CR-06 — Średni — kontrola typów nie jest bramką CI i obecnie zgłasza błędy

**Miejsce:** `pyproject.toml:24`, `.github/workflows/ci.yml:65-103`.

**Problem:** `mypy` jest zależnością developerską, ale workflow go nie uruchamia.
Lokalne `python -m mypy src` zgłasza 26 błędów w 12 plikach. Część to brakujące
stub-y, lecz lista zawiera też rzeczywiste niespójności typów w CLI, kalibracji,
enkoderze fusion, adapterze zewnętrznym, pamięci i pipeline. Ruff jest uruchamiany
wyłącznie jako diagnostyka advisory.

**Wpływ:** deklarowany kontrakt typów ulega erozji, a regresje interfejsów mogą
trafić do głównej gałęzi mimo zielonego CI.

**Wymagana poprawka:** poprawić błędy rzeczywiste, dodać brakujące stub-y lub
precyzyjne lokalne wyłączenia z uzasadnieniem, skonfigurować mypy w
`pyproject.toml`, a następnie uruchamiać go jako obowiązkową bramkę CI. Ruff także
powinien docelowo przestać być advisory.

**Test akceptacyjny:** `python -m mypy src` oraz `python -m ruff check src tests tools`
muszą kończyć się kodem 0 w czystym środowisku developerskim.

## Wykonane kontrole

| Kontrola | Wynik | Uwagi |
| --- | --- | --- |
| `python -m pip install -e .` | ograniczenie środowiska | izolowany build nie mógł pobrać `setuptools>=69` (HTTP 403) |
| `PYTHONPATH=src python -m pytest` | pozytywny | 133 testy przeszły bez instalowania pakietu |
| `python -m ruff check src tests tools` | pozytywny | brak zgłoszeń Ruff |
| `python -m mypy src` | negatywny | 26 błędów w 12 plikach; opisano w CR-06 |
| `python -m mkdocs build --strict` | ograniczenie środowiska | moduł `mkdocs` nie jest zainstalowany |

Pierwsza próba `python -m pytest` bez instalacji ani `PYTHONPATH=src` zakończyła
się 23 błędami importu `s4dtam_benchmark`. Jest to ograniczenie sposobu
uruchomienia projektu w układzie `src`, nie osobny defekt produktu: instrukcja
README i CI poprawnie instalują pakiet przed testami.

## Zalecana kolejność realizacji

1. CR-01 i CR-02 — najpierw zabezpieczyć wiarygodność metryk.
2. CR-03 — ustabilizować pochodzenie wejść przed kolejnymi eksperymentami.
3. CR-04 i CR-05 — usunąć niespójności pipeline'u i diagnostyki.
4. CR-06 — spłacić bieżący dług typów i włączyć stałą bramkę regresji.

Po wdrożeniu poprawek należy ponownie wykonać pełny zestaw testów, statyczne
kontrole oraz smoke benchmark z konfiguracją uruchamianą spoza katalogu repozytorium.
