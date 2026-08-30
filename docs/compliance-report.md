# Raport zgodności wyników

Raportem źródłowym jest `compliance-results.csv` (wersjonowany pusty wzorzec znajduje się
w `docs/templates/`): **jeden wiersz na każdy opublikowany
wynik** (również brak/abort), bez scalania seedów. Kolumny są obowiązkowe:

|Pole|Znaczenie|
|---|---|
|`result_id`|stabilny, unikalny identyfikator|
|`hypothesis`|`H1`–`H7` (wynik pomocniczy nadal dziedziczy hipotezę)|
|`variant`|dokładna nazwa z macierzy ablacji|
|`metric`, `value`, `unit`, `status`|wynik atomowy; `status=missing/abort` wymaga `reason`|
|`dataset`, `dataset_version`, `sequence`, `split`|zamrożone źródło próby; publikowane wyniki muszą mieć `split=test`|
|`seed`|seed treningu/wykonania użyty dla wiersza|
|`hardware_id`|klucz do niezmiennego `hardware.json` (CPU, GPU, RAM, OS, sterownik, power mode)|
|`model_artifact`, `model_sha256`|ID z provenance i skrót dokładnego pliku|
|`config_sha256`, `code_commit`, `run_id`, `timestamp_utc`|pochodzenie wykonania|
|`primary`, `exclusion`, `reason`|rola metryki i jawna decyzja o wykluczeniu|

Przykładowy nagłówek (stanowi normatywną kolejność):

```csv
result_id,hypothesis,variant,metric,value,unit,status,dataset,dataset_version,sequence,split,seed,hardware_id,model_artifact,model_sha256,config_sha256,code_commit,run_id,timestamp_utc,primary,exclusion,reason
```

## Kontrola publikacyjna

Raport jest zgodny tylko, jeśli każdy wiersz łączy się 1:1 z wariantem, provenance modelu,
wersją danych i sprzętem; SHA-256 daje się zweryfikować; nie ma treningu na `test`; wszystkie
pary wariant × seed × sekwencja × metryka są obecne albo mają jawny status. Tabela decyzji
grupuje wyłącznie wcześniej oznaczone metryki główne, pokazuje surowe i Holm-skorygowane
`p`, 95% CI, efekt, licznik braków/wykluczeń i decyzję dla H1–H7. Reviewer podpisuje hash
CSV i tabeli; zagregowana tabela bez źródłowych wierszy nie jest raportem zgodności.

## Reguły transformacji i audytu

`result_id` jest deterministycznym skrótem pól run/variant/dataset/sequence/seed/metric;
duplikat oznacza błąd. Wartość jest zapisywana z pełną precyzją, a zaokrąglenie zachodzi
dopiero w tabeli prezentacyjnej. `NaN`, pusty string i zero nie zastępują braku: używa się
`status=missing` i pozostawia `value` puste. `exclusion=true` nie usuwa wiersza. Jednostki
muszą należeć do specyfikacji metryk, czas być UTC, a hash mieć 64 małe znaki szesnastkowe.

Audyt sprawdza klucze obce do `hardware.json`, manifestu danych i provenance, unikalność,
kompletność iloczynu kartezjańskiego planu, zgodność splitu i hashy oraz to, czy agregaty dają
się odtworzyć wyłącznie z CSV. Następnie niezależny analityk odtwarza tabelę z czystego
środowiska. Raport końcowy podaje wersję schematu, liczbę oczekiwanych/obecnych wierszy,
liczbę braków, abortów i wykluczeń oraz wszystkie naruszenia — także naprawione.
