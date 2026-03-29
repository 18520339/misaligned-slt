# Qualitative Translation Examples

---

**Category:** hallucination | **Reason:** HT hallucination (HT_20)

### Sample: test/06January_2010_Wednesday_tagesschau-7737

**Reference:** es bleibt windig .
**Reference Gloss:** BLEIBEN WIND

| Condition | Gloss Prediction | Translation                                                | sBLEU | Mode   |
| --------- | ---------------- | ---------------------------------------------------------- | ----- | ------ |
| clean     | BLEIBEN WIND     | es bleibt windig .                                         | 1.00  | ACCEPT |
| HT_10     | BLEIBEN WIND     | es bleibt windig .                                         | 1.00  | ACCEPT |
| HT_20     | WIND             | der wind weht schwach bis mäßig im nordosten auch frisch . | 0.04  | HALL   |
| TT_20     | BLEIBEN WIND     | es bleibt windig .                                         | 1.00  | ACCEPT |
| HC_20     | BLEIBEN WIND     | es bleibt windig .                                         | 1.00  | ACCEPT |
| TC_20     | BLEIBEN WIND     | es bleibt windig .                                         | 1.00  | ACCEPT |

---

**Category:** hallucination | **Reason:** HT hallucination (HT_20)

### Sample: test/25November_2010_Thursday_tagesschau-2528

**Reference:** vorsicht wegen straßenglätte .
**Reference Gloss:** VORSICHT POSS-SEIN ACHTUNG STRASSE GLATT

| Condition | Gloss Prediction                     | Translation                            | sBLEU | Mode   |
| --------- | ------------------------------------ | -------------------------------------- | ----- | ------ |
| clean     | VORSICHT ACHTUNG STRASSE GLATT       | vorsicht wegen straßenglätte .         | 1.00  | ACCEPT |
| HT_10     | DIENSTAG STRASSE GLATT               | es muss mit glätte gerechnet werden .  | 0.07  | HALL   |
| HT_20     | IX STRASSE GLATT                     | es kann glatt werden auf den straßen . | 0.06  | HALL   |
| TT_20     | VORSICHT ACHTUNG STRASSE GLATT       | vorsicht wegen straßenglätte .         | 1.00  | ACCEPT |
| HC_20     | REGEN VORSICHT ACHTUNG STRASSE GLATT | vorsicht wegen straßenglätte .         | 1.00  | ACCEPT |
| TC_20     | VORSICHT ACHTUNG STRASSE GLATT ODER  | vorsicht wegen straßenglätte .         | 1.00  | ACCEPT |

---

**Category:** under-generation | **Reason:** TT under-generation (TT_20)

### Sample: test/28March_2011_Monday_tagesschau-2285

**Reference:** im süden gibt es zu beginn der nacht noch wolken die hier und da auch noch ein paar tropfen fallen lassen sonst ist es meist klar oder nur locker bewölkt .
**Reference Gloss:** SUED WOLKE KOMMEN BISSCHEN REGEN KOENNEN REGION KLAR LOCKER WOLKE

| Condition | Gloss Prediction                                             | Translation                                                                      | sBLEU | Mode  |
| --------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------- | ----- | ----- |
| clean     | SUED WOLKE KOMMEN IX KOENNEN REGION KLAR SCHWACH             | im süden ziehen noch ein paar wolkenfelder vorüber sonst ist es meist klar oder  | 0.30  | PART  |
| TT_10     | SUED WOLKE KOMMEN IX KOENNEN REGION KLAR SCHWACH             | im süden ziehen noch ein paar wolkenfelder vorüber sonst ist es meist klar oder  | 0.30  | PART  |
| TT_20     | SUED WOLKE KOMMEN IX KOENNEN REGION KLAR                     | im süden ziehen noch ein paar wolkenfelder vorüber sonst ist es meist klar .     | 0.12  | UNDER |
| HT_20     | IX KOENNEN REGION KLAR SCHWACH                               | hier und da fallen ein paar tropfen sonst ist es meist klar oder nur locker bewö | 0.32  | PART  |
| HC_20     | SUED WOLKE KOMMEN IX KOENNEN REGION KLAR SCHWACH             | im süden ziehen noch ein paar wolkenfelder vorüber sonst ist es meist klar oder  | 0.30  | PART  |
| TC_20     | SUED WOLKE KOMMEN IX KOENNEN REGION KLAR SCHWACH NORDWEST ME | im süden ziehen noch ein paar wolkenfelder vorüber sonst ist es meist klar oder  | 0.34  | PART  |

---

**Category:** mixing | **Reason:** Contamination mixing (HC_20)

### Sample: test/25October_2010_Monday_tagesschau-17

**Reference:** regen und schnee lassen an den alpen in der nacht nach im norden und nordosten fallen hier und da schauer sonst ist das klar .
**Reference Gloss:** REGEN SCHNEE REGION VERSCHWINDEN NORD REGEN KOENNEN REGION STERN KOENNEN SEHEN

| Condition | Gloss Prediction                                             | Translation                                                                      | sBLEU | Mode |
| --------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------- | ----- | ---- |
| clean     | REGEN SCHNEE ALPEN VERSCHWINDEN NORD REGEN KOENNEN REGION ST | die regenwolken lassen im laufe des tages nach am alpenrand schwächt sich ein we | 0.09  | HALL |
| HC_10     | REGEN SCHNEE ALPEN VERSCHWINDEN NORDOST NORD REGEN KOENNEN R | in der südhälfte lässt der regen und schnee am alpenrand nach im nordosten regne | 0.09  | HALL |
| HC_20     | REGEN SCHNEE ALPEN VERSCHWINDEN NORD REGEN KOENNEN REGION ST | im übrigen land lassen die regen und schneefälle am alpenrand nach im nordosten  | 0.05  | HALL |
| HT_20     | ALPEN VERSCHWINDEN NORD REGEN KOENNEN REGION STERN KOENNEN S | am tag lässt der regen von den alpen nach am tag regnet es im norden und osten h | 0.11  | HALL |
| TT_20     | REGEN SCHNEE ALPEN VERSCHWINDEN NORD REGEN KOENNEN REGION    | die regen und schneefälle lassen am alpenrand langsam nach im norden und nordost | 0.18  | HALL |
| TC_20     | REGEN SCHNEE ALPEN VERSCHWINDEN NORD REGEN KOENNEN REGION ST | am donnerstag regnet oder schneit es an den alpen noch länger anhaltend im norde | 0.13  | HALL |

---

**Category:** repetition | **Reason:** Repetition (TT_20)

### Sample: test/26April_2010_Monday_tagesschau-7272

**Reference:** die schauer und gewitter im norden klingen in der nacht ab an den alpen fallen noch ein paar regentropfen stellenweise bildet sich nebel .
**Reference Gloss:** NORD REGEN NACHT VERSCHWINDEN ALPEN REGEN KOENNEN IX NEBEL

| Condition | Gloss Prediction                                             | Translation                                                                      | sBLEU | Mode  |
| --------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------- | ----- | ----- |
| clean     | NORD REGEN WOLKE VERSCHWINDEN ALPEN REGEN KOENNEN IX NEBEL   | im norden regnet es heute nacht ab und an an an den alpen regnet es stellenweise | 0.18  | REPET |
| TT_10     | NORD REGEN ABEND VERSCHWINDEN ALPEN REGEN KOENNEN IX NEBEL   | im norden regnet es in der nacht ab und an an an den alpen regnet es stellenweis | 0.26  | REPET |
| TT_20     | NORD REGEN ABEND VERSCHWINDEN ALPEN REGEN KOENNEN            | im norden regnet es in der nacht ab und an an an den alpen regnet es mitunter no | 0.17  | REPET |
| HT_20     | NACHT VERSCHWINDEN ALPEN REGEN KOENNEN IX NEBEL              | in der nacht klart es gebietsweise auf an den alpen regnet es stellenweise etwas | 0.19  | PART  |
| HC_20     | TAG TEMPERATUR NORD REGEN WOLKE VERSCHWINDEN ALPEN REGEN KOE | in der nacht klart es im norden zum teil auf an den alpen regnet es vereinzelt e | 0.20  | PART  |
| TC_20     | NORD REGEN WOLKE VERSCHWINDEN ALPEN REGEN KOENNEN IX NEBEL J | in der nordhälfte verläuft die nacht klar an den alpen regnet es ab und an stell | 0.23  | PART  |

---

**Category:** compound | **Reason:** Compound degradation (HT_05+TT_05)

### Sample: test/28October_2009_Wednesday_tagesschau-4540

**Reference:** im norden und nordosten bleibt es meist bedeckt mitunter fällt dort etwas regen .
**Reference Gloss:** NORDOST BEWÖLKT IX REGEN

| Condition | Gloss Prediction                | Translation                                                                      | sBLEU | Mode   |
| --------- | ------------------------------- | -------------------------------------------------------------------------------- | ----- | ------ |
| clean     | NORDOST NEBEL IX REGEN          | im norden und nordosten bleibt es meist stark bewölkt hier und da fällt regen .  | 0.45  | ACCEPT |
| HT_10     | NORDOST MEISTENS WOLKE IX REGEN | im nordosten ist es heute nacht meist stark bewölkt hier und da fällt regen .    | 0.07  | INCOH  |
| HT_20     | NORD MEISTENS WOLKE IX REGEN    | im norden bleibt es meist stark bewölkt hier und da fällt regen .                | 0.17  | PART   |
| TT_20     | NORDOST MEISTENS NEBEL IX       | im norden und nordosten bleibt es meist stark bewölkt .                          | 0.44  | ACCEPT |
| HC_20     | SONNE NORD NEBEL IX REGEN       | die sonne zeigt sich nur selten am ehesten im norden und osten ist es stark bewö | 0.07  | HALL   |
| TC_20     | NORDOST NEBEL IX REGEN ORT      | im norden und nordosten bleibt es meist stark bewölkt hier und da fällt etwas re | 0.47  | ACCEPT |
