# Prerejestracja badania porównawczego S4D-TAM

**Stan:** protokół prospektywny, zamrożony przed ujawnieniem zbioru testowego

**Wersja:** 1.1 · **data zamrożenia:** 2026-08-30 · **jednostka inferencji:** misja/sekwencja

## Cel, zakres i zasada estymacji

Celem badania jest oszacowanie przyczynowego wkładu siedmiu komponentów S4D-TAM do
dokładności lokalizacji, prognozowania, bezpieczeństwa nawigacji i kosztu obliczeniowego.
Każdy kontrast porównuje model pełny z modelem różniącym się dokładnie jednym komponentem.
Architektura poza wskazanym przełącznikiem, budżet optymalizacji, dane, seed, harmonogram
degradacji czujników i procedura oceny pozostają identyczne. Estymandem jest średnia
sparowana różnica na populacji scenariuszy objętych zamrożonymi manifestami; w przypadku
mission success jest nim różnica prawdopodobieństw sukcesu.

Dokument nie jest dowodem rejestracji w zewnętrznym repozytorium. Przed rozpoczęciem
badania należy zdeponować jego hash SHA-256, commit i znacznik czasu. Każda późniejsza
zmiana otrzymuje aneks; analiza nieprzewidziana poniżej jest oznaczana jako eksploracyjna.

## Hipotezy i wyniki

Każde porównanie jest sparowane względem wariantu `full`, na tych samych sekwencjach i
seedach. Kierunki „lepiej” oznaczają mniejszą wartość ATE/kolizji/latencji, a większą dla
IoU i powodzenia misji.

|ID|Jedyna zmiana wariantu|Hipoteza|Metryka główna|Metryki pomocnicze|
|---|---|---|---|---|
|H1|`H1_no_semantics`: semantyka wyłączona|Semantyka poprawia nawigację w obiektach dynamicznych.|mission success|collisions/km, semantic mIoU|
|H2|`H2_no_temporal_state`: stan czasowy wyłączony|Stan czasowy poprawia prognozę 1 s.|forecast 1 s IoU|flow EPE, temporal flip rate|
|H3|`H3_no_calibrated_uncertainty`: kalibracja wyłączona|Kalibrowana niepewność ogranicza ryzyko.|collisions/km|ECE, NLL, near misses|
|H4|`H4_no_topology`: topologia wyłączona|Topologia zwiększa powodzenie misji.|mission success|path efficiency, ATE RMSE|
|H5|`H5_no_reference_map`: mapa referencyjna wyłączona|Mapa referencyjna zmniejsza błąd lokalizacji.|ATE RMSE [m]|RPE, final drift %|
|H6|`H6_no_risk_prediction`: predykcja ryzyka wyłączona|Predykcja ryzyka zmniejsza kolizje.|collisions/km|near misses, clearance|
|H7|`H7_no_token_lifecycle`: cykl tokenów wyłączony|Cykl tokenów zmniejsza koszt bez utraty jakości.|latency p95 [ms]|map bytes, peak RSS, mission success (non-inferiority)|

H1–H6 są testami wyższości w kierunku przewidzianym w tabeli. H7 ma dwa współgłówne
warunki: niższa latency p95 oraz non-inferiority mission success z marginesem 2 p.p.; H7
zostaje potwierdzona wyłącznie po spełnieniu obu. Testy są dwustronne przy rodzinnym
α=0,05, z wyjątkiem jednostronnego testu non-inferiority. Metryki pomocnicze nie zmieniają
decyzji potwierdzającej.

## Zbiory, podział i wykluczenia

- Zamrożone wersje z manifestami SHA-256: TartanAir, Blackbird, MARSIM i AeroVerse.
  Identyfikator `frozen-2026-01` jest nazwą kohorty, nie wersją dostawcy; raport musi podać
  wersję źródłową i hash każdego pliku. `train` służy do treningu, `calibration` do progów,
  wczesnego zatrzymania i kalibracji,
  a `test` wyłącznie do końcowej oceny. Jednostką próby jest sekwencja/misja.
- Offline: po 20 uprzednio wylosowanych sekwencji testowych na zbiór (80). SIL: 30
  sparowanych scenariuszy × 5 seedów. HIL: 20 scenariuszy × 5 seedów. Real-flight:
  20 sparowanych misji na wariant, o ile bramka bezpieczeństwa pozostaje otwarta.
- Przed ujawnieniem wyniku wyklucza się tylko: niezgodność sumy kontrolnej, brak ponad
  5% wymaganych znaczników czasu, uszkodzenie czujnika potwierdzone logiem, interwencję
  regulatora przestrzeni lub pogodę poza limitami protokołu. Kolizje, timeouty, aborty
  algorytmu i nieudane misje pozostają w mianowniku. Wykluczenie zatwierdzają dwie osoby,
  z kodem przyczyny; nie zastępuje się próby po obejrzeniu wyników.

## Randomizacja, zaślepienie i kontrola wykonania

Kolejność wariantów w każdym bloku dataset × scenario × seed jest wyznaczana raz przez
permutację z generatorem PCG64 i archiwizowana przed uruchomieniem. Pary korzystają z tych
samych zakłóceń i stanu początkowego. Osoba zatwierdzająca wykluczenia widzi log techniczny,
lecz nie nazwę wariantu ani metryki. Analiza potwierdzająca jest uruchamiana jednokrotnie
przez skrypt na zamrożonym pliku wierszowym; autor modelu nie może ręcznie usuwać prób.

## Liczebność, model statystyczny i dane brakujące

Wielkości ustalono przed testem dla mocy 0,8 i α=0,05: 80 par wykrywa standaryzowany efekt
około 0,32 w metrykach ciągłych (przy założeniach sparowanego testu t); ta wartość jest
uzasadnieniem projektowym, nie gwarancją mocy po korekcji. Próby SIL/HIL zwiększają
precyzję estymacji awarii, natomiast 20 lotów jest etapem bezpieczeństwa i wykonalności,
nie badaniem równoważności małych efektów.

Dla metryk ciągłych raportujemy sparowaną różnicę średnich i BCa bootstrap 95% CI z 10 000
resamplowań na poziomie jednostki inferencji. Mission success analizujemy modelem logistycznym
z efektem scenariusza, a zdarzenia na dystans — modelem Poissona lub ujemno-dwumianowym
z logarytmem dystansu jako offsetem (wybór na podstawie prerejestrowanego testu nadmiernej
dyspersji). Raport zawiera także surowe dane, efekt, standard error i surowe `p`.

Siedem decyzji H1–H7 tworzy jedną rodzinę i podlega sekwencyjnej korekcji Holma według
rosnących `p`; remisy rozstrzyga numer hipotezy. Wyniki pomocnicze mają CI i oddzielną,
opisową korekcję Holm, bez wpływu na decyzję. Braki nie są imputowane. Awaria/abort jest
wynikiem bezpieczeństwa i pozostaje w mianowniku; brak techniczny raportuje się wraz z
przyczyną, a analizę complete-case uzupełnia prerejestrowana analiza worst-case.

## Kryterium decyzji i raportowanie

Hipotezę oznacza się jako „wspieraną”, gdy skorygowane `p < 0,05`, 95% CI wyklucza zero
w przewidzianym kierunku i nie wystąpiło naruszenie integralności danych. Brak istotności
nie jest dowodem braku efektu. Publikowane są wszystkie hipotezy, seedy, awarie, wykluczenia,
przedziały oraz wyniki sprzeczne z oczekiwaniem. Schemat danych źródłowych definiuje raport
zgodności, a szczegóły obliczeń — `docs/methodology.md`.

Konfigurację wykonawczą stanowi `configs/experiments/ablation.yaml`; walidator musi
zaakceptować ją przed uruchomieniem.
