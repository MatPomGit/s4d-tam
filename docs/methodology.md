# Metodologia eksperymentalna i plan replikacji

## Dwa poziomy porównania

Walidacja S4D-TAM jest rozdzielona na dwa odrębne poziomy, ponieważ odpowiadają one na różne pytania badawcze i mają inne estymandy.

**Porównanie zewnętrzne** ocenia konkurencyjność całego systemu S4D-TAM względem niezależnych implementacji ORB-SLAM3, VINS-Mono, FAST-LIO2 i LIO-SAM. Wszystkie systemy są oceniane przez ten sam kontrakt danych i te same ewaluatory, ale nie są częścią rodziny H1-H7.

**Porównanie wewnętrzne** jest badaniem mechanizmu. Model `full` jest porównywany wyłącznie z siedmioma wariantami H1-H7, z których każdy wyłącza dokładnie jeden komponent S4D-TAM. Ten poziom podlega zamrożonej prerejestracji i korekcji wielokrotnych porównań.

Konfiguracje są jawnie oznaczane przez `comparison_level: external` albo `comparison_level: internal` i walidowane przez `s4dtam-bench validate-comparison`. Szczegóły znajdują się w [Two-level comparison protocol](comparison-protocol.md).

## Projekt badania wewnętrznego

Badanie H1-H7 ma układ blokowy, sparowany i wielozbiorowy. Blok stanowi kombinacja zbioru, scenariusza, seeda oraz profilu degradacji GNSS. Wewnątrz bloku wykonywane są `full` i wszystkie warianty H1-H7. Taki projekt kontroluje trudność scenariusza i realizację szumu; nie upoważnia jednak do uogólnienia poza populację warunków opisaną w manifestach.

Jedyną interwencją w kontrastach jest przełącznik komponentu. Ponowne dostrojenie wyłącznie wybranego wariantu, nierówny budżet obliczeń albo użycie innego preprocessingu stanowi konfundowanie i unieważnia kontrast. Jeżeli usunięcie komponentu wymaga mechanicznej zmiany wymiaru, adapter zachowuje liczbę parametrów przez neutralny moduł bez uczonych informacji, a odstępstwo jest rejestrowane.

## Projekt porównania zewnętrznego

S4D-TAM oraz każdy baseline są uruchamiane na identycznych zamrożonych sekwencjach. Każdy baseline musi mieć przypięty commit źródłowy, środowisko lub digest kontenera, kalibrację, mapowanie tematów ROS, politykę loop closure, politykę warm-up oraz sprzęt. Wynik jest przyjmowany do wspólnego ewaluatora dopiero po przejściu walidacji kontraktu `AlgorithmResult`.

Porównanie zewnętrzne nie jest ablacją. Różnica względem ORB-SLAM3, VINS-Mono, FAST-LIO2 lub LIO-SAM jest interpretowana jako różnica między kompletnymi systemami przy zdefiniowanych warunkach sensorowych, a nie jako przyczynowy efekt konkretnego modułu S4D-TAM. Dodatkowe systemy mogą zostać włączone wyłącznie przed zamrożeniem badania zewnętrznego i po spełnieniu kryteriów zgodności sensorowej oraz reprodukowalności.

## Populacje danych i zapobieganie przeciekowi

Splity tworzy się na poziomie środowiska/lotu, nigdy sąsiednich klatek. `train` jest jedynym źródłem aktualizacji wag; `calibration` służy do hiperparametrów, progów, normalizacji i kalibracji niepewności; `test` pozostaje niedostępny aż do zamrożenia kodu. Artefakt zawiera SHA-256, konfigurację treningu, wersję kodu, listę identyfikatorów danych i splitów oraz seed. Walidator odrzuca artefakt wspólny dla dwóch ablacji i każdy styk zbioru treningowego ze splitem ewaluacyjnym. Ostateczny audyt porównuje identyfikatory próbek, nie tylko nazwy splitów.

## Standaryzacja pomiaru

- Lokalizacja: wspólne znaczniki czasu, interpolacja wyłącznie w dopuszczalnej tolerancji, transformacja SE(3) bez skali, metry na poziomie sekwencji.
- Prognoza: wspólna siatka, maska obserwowalności i horyzont liczony od czasu predykcji; puste unie są raportowane według jednej zamrożonej konwencji, nie pomijane selektywnie.
- Nawigacja: sukces wymaga osiągnięcia celu w czasie bez kolizji, geofence i interwencji; dystans ekspozycji pochodzi z ground truth, a abort liczy się jako niepowodzenie.
- Wydajność: identyczny sprzęt i power mode, trzy rozgrzewki wyłączone z estymacji, co najmniej pięć mierzonych powtórzeń; latency obejmuje preprocessing, inferencję i map update.

Jednostką analizy pozostaje sekwencja/misja. Klatki nie są traktowane jako niezależne próby. Definicje metryk, kierunki i jednostki są w `docs/metrics.md`; algorytm statystyczny badania wewnętrznego jest zamrożony w prerejestracji.

## Procedura wykonawcza

1. Zweryfikować podpis źródeł, manifesty SHA-256, licencje i brak nakładania identyfikatorów.
2. Zbudować offline środowisko z wheelhouse, zapisać obraz/sterowniki/sprzęt i uruchomić testy.
3. Zweryfikować poziom porównania przez `validate-comparison`; dla badania H1-H7 dodatkowo zweryfikować macierz ablacji.
4. Zapisać hash efektywnej konfiguracji i wygenerować run ID.
5. Wykonać systemy lub warianty w zamrożonej kolejności randomizowanej; stdout, stderr i telemetry są append-only. Proces nie może nadpisywać artefaktu modelu ani manifestu danych.
6. Zmaterializować jeden wiersz na wynik, przeprowadzić audyt kompletności, a dopiero potem odślepić warianty i wykonać właściwą analizę statystyczną.
7. Opublikować surowe dozwolone dane, braki, awarie, kod analizy, środowisko, provenance, raport incydentów po anonimizacji oraz wynik `sha256sum -c`.

### Polityka częściowych awarii

Runner izoluje awarie na poziomie algorytmu. Błąd kalibracji oznacza algorytm jako
`calibration_failed`: nie jest on uruchamiany na żadnej sekwencji ewaluacyjnej, ale
kalibracja i wykonanie pozostałych algorytmów trwają dalej. Analogicznie błąd podczas
wykonania oznacza algorytm jako `execution_failed` i wyłącza jego kolejne uruchomienia.
Każda awaria trafia do `failures.json` z fazą, nazwą algorytmu, typem wyjątku,
komunikatem oraz identyfikatorem danych kalibracyjnych; dla awarii wykonania raport
zawiera też zbiór i sekwencję. Raporty JSON są publikowane atomowo. Jeżeli nie powstał
żaden poprawny rekord metryki, runner najpierw zapisuje `failures.json`, a dopiero
potem zgłasza końcowy `RuntimeError`. Wyniki algorytmów, które zakończyły pracę
poprawnie, pozostają raportowane jako wynik częściowy i muszą być interpretowane wraz
z rejestrem awarii.

## Replikacja i granice wnioskowania

Replikator nie odtwarza lokalnego środowiska autora, lecz buduje je z dostarczonych wheelów przy wyłączonej sieci. Replikacja jest ścisła, gdy zgadzają się hashe wejść i wyniki deterministyczne; jest obliczeniowa, gdy wartości mieszczą się w z góry ustalonej tolerancji numerycznej; jest inferencyjna, gdy kierunek i decyzja pozostają zgodne mimo innego sprzętu. Każdy poziom raportuje się osobno. Wyniki symulacyjne nie dowodzą bezpieczeństwa lotniczego, a real-flight pozostaje ograniczony do zatwierdzonej obwiedni operacyjnej.
