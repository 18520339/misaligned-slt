# Qualitative Translation Examples

---
**Category:** hallucination | **Reason:** HT hallucination (HT_20)

### Sample: test/06January_2010_Wednesday_tagesschau-7737

**Reference:** es bleibt windig .

**Reference Gloss:** BLEIBEN WIND

| Condition | Gloss Prediction | Translation | sBLEU | Mode |
|-----------|------------------|-------------|-------|------|
| clean | BLEIBEN WIND | es bleibt windig . | 100.00 | ACCEPT |
| HT_10 | BLEIBEN WIND | es bleibt windig . | 100.00 | ACCEPT |
| HT_20 | WIND | der wind weht schwach bis mÃ¤ÃŸig im nordosten auch frisch . | 1.89 | HALL |
| TT_20 | BLEIBEN WIND | es bleibt windig . | 100.00 | ACCEPT |
| HC_20 | BLEIBEN WIND | es bleibt windig . | 100.00 | ACCEPT |
| TC_20 | BLEIBEN WIND | es bleibt windig . | 100.00 | ACCEPT |

---
**Category:** hallucination | **Reason:** HT hallucination (HT_20)

### Sample: test/20April_2011_Wednesday_tagesschau-3086

**Reference:** auch am tag wieder viel sonnenschein spÃ¤ter bilden sich hier und da ein paar quellwolken .

**Reference Gloss:** AUCH TAG VIEL SONNE SPAETER IX QUELL WOLKE

| Condition | Gloss Prediction | Translation | sBLEU | Mode |
|-----------|------------------|-------------|-------|------|
| clean | AUCH TAG VIEL SONNE IM-VERLAUF IX WOLKE | auch am tag viel sonne im laufe des tages bilden sich spÃ¤ter nur hier und da ei | 40.16 | ACCEPT |
| HT_10 | VIEL SONNE IM-VERLAUF IX WOLKE | viel sonnenschein gibt es dafÃ¼r im sÃ¼den spÃ¤ter auch dort wo sich spÃ¤ter die | 5.24 | HALL |
| HT_20 | SONNE IM-VERLAUF IX WOLKE | die sonne zeigt sich nur selten spÃ¤ter werden die wolken dichter . | 1.61 | HALL |
| TT_20 | AUCH TAG VIEL SONNE IM-VERLAUF IX | auch am tag viel sonnenschein erst scheint die sonne spÃ¤ter nur hier und da ver | 11.15 | PART |
| HC_20 | IX AUCH TAG VIEL SONNE SPAETER IX WOLKE | auch am tag scheint verbreitet die sonne spÃ¤ter bilden sich nur noch hier und d | 44.48 | ACCEPT |
| TC_20 | AUCH TAG VIEL SONNE IM-VERLAUF IX WOLKE MITTWOCH | auch am tag viel sonne im laufe des tages entwickeln sich spÃ¤ter nur hier und d | 38.28 | PART |

---
**Category:** under-generation | **Reason:** TT under-generation (TT_20)

### Sample: test/12August_2010_Thursday_tagesschau-4403

**Reference:** quer Ã¼ber deutschland liegt eine luftmassengrenze die kÃ¼hlere meeresluft von heiÃŸer luft aus dem sÃ¼dosten trennt .

**Reference Gloss:** DEUTSCH LAND KOMMEN KUEHL HEISS GETRENNT WARMFRONT

| Condition | Gloss Prediction | Translation | sBLEU | Mode |
|-----------|------------------|-------------|-------|------|
| clean | DEUTSCH LAND KOMMEN KUEHL HEISS KOMMEN | deutschland liegt weiter zwischen einem atlantikhoch und einer luftmassengrenze  | 2.64 | HALL |
| TT_10 | DEUTSCH LAND KOMMEN KUEHL HEISS KOMMEN | deutschland liegt weiter zwischen einem atlantikhoch und einer luftmassengrenze  | 2.64 | HALL |
| TT_20 | DEUTSCH LAND KOMMEN KUEHL HEISS | es lenkt kÃ¼hlere luft nach deutschland . | 1.12 | UNDER |
| HT_20 | KOMMEN KUEHL HEISS | es lenkt kÃ¼hlere atlantikluft nach deutschland . | 1.04 | HALL |
| HC_20 | ORT REGEN DEUTSCH LAND KOMMEN KUEHL HEISS KOMMEN | es lenkt regenwolken nach deutschland und lenkt kÃ¼hlere meeresluft nach deutsch | 2.82 | HALL |
| TC_20 | DEUTSCH LAND KOMMEN KUEHL HEISS KOMMEN DEUTSCH LAND | es lenkt kÃ¼hlere luft aus westen nach deutschland . | 2.62 | INCOH |

---
**Category:** under-generation | **Reason:** TT under-generation (TT_20)

### Sample: test/19February_2010_Friday_tagesschau-4093

**Reference:** da die luftdruckunterschiede nahe des tiefzentrums groÃŸ sind wird es im kÃ¼stenumfeld morgen teilweise stÃ¼rmisch .

**Reference Gloss:** NAH TIEF LUFT DRUCK STARK UNTERSCHIED DESHALB TEIL WIND

| Condition | Gloss Prediction | Translation | sBLEU | Mode |
|-----------|------------------|-------------|-------|------|
| clean | NAH TIEF LUFT DRUCK HAUPTSÃ„CHLICH DESHALB TEIL WIND | tiefer luftdruck bestimmt in den kommenden tagen in weiten teilen deutschlands u | 1.28 | HALL |
| TT_10 | NAH TIEF LUFT DRUCK HAUPTSÃ„CHLICH DESHALB TEIL | in den kommenden tagen bestimmt tiefer luftdruck unser wetter und in einigen reg | 1.21 | HALL |
| TT_20 | NAH TIEF LUFT DRUCK HAUPTSÃ„CHLICH TEMPERATUR DESHALB | tiefer luftdruck bestimmt weiterhin unser wetter . | 0.91 | HALL |
| HT_20 | TIEF LUFT DRUCK HAUPTSÃ„CHLICH DESHALB TEIL WIND | tiefer luftdruckunterschiede sorgen teilweise fÃ¼r windiges und sehr windiges we | 1.57 | HALL |
| HC_20 | STURM NAH TIEF LUFT DRUCK ÃœBERWIEGEND DESHALB TEIL WIND | ein sturmtief bringt uns krÃ¤ftige tiefdruckgebiete und teilweise windiges wette | 1.42 | HALL |
| TC_20 | NAH TIEF LUFT DRUCK ÃœBERWIEGEND DESHALB TEIL WIND NEU | in den kommenden tagen bestimmt tiefer luftdruck unser wetter und teilweise auch | 1.47 | HALL |

---
**Category:** mixing | **Reason:** Contamination mixing (HC_20)

### Sample: test/25October_2010_Monday_tagesschau-17

**Reference:** regen und schnee lassen an den alpen in der nacht nach im norden und nordosten fallen hier und da schauer sonst ist das klar .

**Reference Gloss:** REGEN SCHNEE REGION VERSCHWINDEN NORD REGEN KOENNEN REGION STERN KOENNEN SEHEN

| Condition | Gloss Prediction | Translation | sBLEU | Mode |
|-----------|------------------|-------------|-------|------|
| clean | REGEN SCHNEE ALPEN VERSCHWINDEN NORD REGEN KOENNEN REGION ST | die regenwolken lassen im laufe des tages nach am alpenrand schwÃ¤cht sich ein w | 9.24 | HALL |
| HC_10 | REGEN SCHNEE ALPEN VERSCHWINDEN NORDOST NORD REGEN KOENNEN R | in der sÃ¼dhÃ¤lfte lÃ¤sst der regen und schnee am alpenrand nach im nordosten re | 6.04 | HALL |
| HC_20 | REGEN SCHNEE ALPEN VERSCHWINDEN NORD REGEN KOENNEN REGION ST | im laufe des tages lassen die regen und schneefÃ¤lle am alpenrand nach im norden | 16.34 | HALL |
| HT_20 | ALPEN VERSCHWINDEN NORD REGEN KOENNEN REGION STERN KOENNEN S | am tag lÃ¤sst der regen von den alpen nach am tag regnet es im norden und osten  | 7.45 | HALL |
| TT_20 | REGEN SCHNEE ALPEN VERSCHWINDEN NORD REGEN KOENNEN REGION | die regen und schneefÃ¤lle lassen am alpenrand langsam nach im norden und nordos | 17.73 | HALL |
| TC_20 | REGEN SCHNEE ALPEN VERSCHWINDEN NORD REGEN KOENNEN REGION ST | am donnerstag regnet oder schneit es an den alpen noch lÃ¤nger anhaltend im nord | 13.33 | HALL |

---
**Category:** repetition | **Reason:** Repetition (TT_20)

### Sample: test/26April_2010_Monday_tagesschau-7272

**Reference:** die schauer und gewitter im norden klingen in der nacht ab an den alpen fallen noch ein paar regentropfen stellenweise bildet sich nebel .

**Reference Gloss:** NORD REGEN NACHT VERSCHWINDEN ALPEN REGEN KOENNEN IX NEBEL

| Condition | Gloss Prediction | Translation | sBLEU | Mode |
|-----------|------------------|-------------|-------|------|
| clean | NORD REGEN WOLKE VERSCHWINDEN ALPEN REGEN KOENNEN IX NEBEL | im norden regnet es heute nacht ab und an an an den alpen regnet es stellenweise | 18.09 | REPET |
| TT_10 | NORD REGEN ABEND VERSCHWINDEN ALPEN REGEN KOENNEN IX NEBEL | im norden regnet es in der nacht ab und an an an den alpen regnet es stellenweis | 25.82 | REPET |
| TT_20 | NORD REGEN ABEND VERSCHWINDEN ALPEN REGEN KOENNEN | im norden regnet es in der nacht ab und an an an den alpen regnet es mitunter no | 17.00 | REPET |
| HT_20 | NACHT VERSCHWINDEN ALPEN REGEN KOENNEN IX NEBEL | in der nacht klart es gebietsweise auf an den alpen regnet es stellenweise etwas | 19.06 | PART |
| HC_20 | TAG TEMPERATUR NORD REGEN WOLKE VERSCHWINDEN ALPEN REGEN KOE | in der nacht klart es im norden zum teil auf an den alpen regnet es vereinzelt e | 20.10 | PART |
| TC_20 | NORD REGEN WOLKE VERSCHWINDEN ALPEN REGEN KOENNEN IX NEBEL J | in der nordhÃ¤lfte verlÃ¤uft die nacht klar an den alpen regnet es ab und an ste | 23.46 | PART |

---
**Category:** compound | **Reason:** Compound degradation (HT_05+TT_05)

### Sample: test/21August_2011_Sunday_tagesschau-263

**Reference:** in der nordhÃ¤lfte gibt es zu beginn der nacht noch einzelne heftige schauer und gewitter dann klar dass dort teilweise auf .

**Reference Gloss:** NORD NACHT ANFANG ENORM SCHAUER GEWITTER DANN WOLKE VERSCHWINDEN

| Condition | Gloss Prediction | Translation | sBLEU | Mode |
|-----------|------------------|-------------|-------|------|
| clean | NORD NACHT ANFANG ENORM SCHAUER UND GEWITTER DANN WOLKE VERS | in der nordhÃ¤lfte beginnt die nacht mit teilweise krÃ¤ftigen schauern und gewit | 5.32 | HALL |
| HT_10 | NORD NACHT ANFANG ENORM SCHAUER GEWITTER DANN WOLKE VERSCHWI | in der nordhÃ¤lfte beginnt die nacht mit teilweise krÃ¤ftigen schauern und gewit | 5.32 | HALL |
| HT_20 | NACHT ANFANG ENORM SCHAUER GEWITTER DANN WOLKE VERSCHWINDEN | anfangs noch teilweise krÃ¤ftige schauer und gewitter die aber abklingen . | 4.08 | INCOH |
| TT_20 | NORD NACHT ANFANG ENORM SCHAUER UND GEWITTER DANN | an der nordsee beginnt die nacht mit teilweise krÃ¤ftigen schauern und gewittern | 1.16 | HALL |
| HC_20 | ELF GRAD NORD NACHT ANFANG ENORM SCHAUER GEWITTER DANN WOLKE | am tag elf grad bei schauern im norden anfangs noch teilweise krÃ¤ftige gewitter | 1.21 | HALL |
| TC_20 | NORD NACHT ANFANG ENORM SCHAUER UND GEWITTER DANN WOLKE VERS | in der nordhÃ¤lfte beginnt die nacht mit teilweise krÃ¤ftigen schauern auch mit  | 5.07 | HALL |
