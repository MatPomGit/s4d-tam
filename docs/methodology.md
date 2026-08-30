# Metodologia eksperymentalna i plan replikacji

## Projekt badania

Badanie ma układ blokowy, sparowany i wielozbiorowy. Blok stanowi kombinacja zbioru,
scenariusza, seeda oraz profilu degradacji GNSS. Wewnątrz bloku wykonywane są `full` i
wszystkie warianty H1–H7. Taki projekt kontroluje trudność scenariusza i realizację szumu;
nie upoważnia jednak do uogólnienia poza populację warunków opisaną w manifestach.

Jedyną interwencją w kontrastach jest przełącznik komponentu. Ponowne dostrojenie wyłącznie
wybranego wariantu, nierówny budżet obliczeń albo użycie innego preprocessingu stanowi
konfundowanie i unieważnia kontrast. Jeżeli usunięcie komponentu wymaga mechanicznej zmiany
wymiaru, adapter zachowuje liczbę parametrów przez neutralny moduł bez uczonych informacji,
a odstępstwo jest rejestrowane.

## Populacje danych i zapobieganie przeciekowi

Splity tworzy się na poziomie środowiska/lotu, nigdy sąsiednich klatek. `train` jest jedynym
źródłem aktualizacji wag; `calibration` służy do hiperparametrów, progów, normalizacji i
kalibracji niepewności; `test` pozostaje niedostępny aż do zamrożenia kodu. Artefakt zawiera
SHA-256, konfigurację treningu, wersję kodu, listę identyfikatorów danych i splitów oraz seed.
Walidator odrzuca artefakt wspólny dla dwóch ablacji i każdy styk zbioru treningowego ze
splitem ewaluacyjnym. Ostateczny audyt porównuje identyfikatory próbek, nie tylko nazwy splitów.

## Standaryzacja pomiaru

- Lokalizacja: wspólne znaczniki czasu, interpolacja wyłącznie w dopuszczalnej tolerancji,
  transformacja SE(3) bez skali, metry na poziomie sekwencji.
- Prognoza: wspólna siatka, maska obserwowalności i horyzont liczony od czasu predykcji;
  puste unie są raportowane według jednej zamrożonej konwencji, nie pomijane selektywnie.
- Nawigacja: sukces wymaga osiągnięcia celu w czasie bez kolizji, geofence i interwencji;
  dystans ekspozycji pochodzi z ground truth, a abort liczy się jako niepowodzenie.
- Wydajność: identyczny sprzęt i power mode, trzy rozgrzewki wyłączone z estymacji, co
  najmniej pięć mierzonych powtórzeń; latency obejmuje preprocessing, inferencję i map update.

Jednostką analizy pozostaje sekwencja/misja. Klatki nie są traktowane jako niezależne próby.
Definicje metryk, kierunki i jednostki są w `docs/metrics.md`; algorytm statystyczny jest
zamrożony w prerejestracji.

## Procedura wykonawcza

1. Zweryfikować podpis źródeł, manifesty SHA-256, licencje i brak nakładania identyfikatorów.
2. Zbudować offline środowisko z wheelhouse, zapisać obraz/sterowniki/sprzęt i uruchomić testy.
3. Uruchomić walidator macierzy, zapisać hash efektywnej konfiguracji i wygenerować run ID.
4. Wykonać warianty w zamrożonej kolejności randomizowanej; stdout, stderr i telemetry są
   append-only. Proces nie może nadpisywać artefaktu modelu ani manifestu danych.
5. Zmaterializować jeden wiersz na wynik, przeprowadzić audyt kompletności, a dopiero potem
   odślepić warianty i wykonać analizę potwierdzającą.
6. Opublikować surowe dozwolone dane, braki, awarie, kod analizy, środowisko, provenance,
   raport incydentów po anonimizacji oraz wynik `sha256sum -c`.

## Replikacja i granice wnioskowania

Replikator nie odtwarza lokalnego środowiska autora, lecz buduje je z dostarczonych wheelów
przy wyłączonej sieci. Replikacja jest ścisła, gdy zgadzają się hashe wejść i wyniki
deterministyczne; jest obliczeniowa, gdy wartości mieszczą się w z góry ustalonej tolerancji
numerycznej; jest inferencyjna, gdy kierunek i decyzja pozostają zgodne mimo innego sprzętu.
Każdy poziom raportuje się osobno. Wyniki symulacyjne nie dowodzą bezpieczeństwa lotniczego,
a real-flight pozostaje ograniczony do zatwierdzonej obwiedni operacyjnej.
