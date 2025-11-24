needed <- c("rvest","stringr","dplyr","tidyverse","readr","tidytext","SnowballC","wordcloud","topicmodels","e1071","caret","tibble","tidyr","ggplot2")
to_install <- setdiff(needed, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

library(rvest)
library(stringr)
library(dplyr)
library(tidyverse)
library(readr)
library(tidytext)
library(SnowballC)
library(wordcloud)
library(topicmodels)
library(e1071)
library(caret)
library(tibble)
library(tidyr)
library(ggplot2)


# 1. Data Collection & Cleaning
urls <- c(
  "https://www.metacritic.com/tv/foundation/user-reviews/?filter=Negative%20Reviews",
  "https://www.metacritic.com/tv/bridgerton/user-reviews/?filter=Negative%20Reviews",
  "https://www.metacritic.com/tv/alien-earth/user-reviews/?filter=Negative%20Reviews",
  "https://www.metacritic.com/tv/wednesday/user-reviews/?filter=Negative%20Reviews",
  "https://www.metacritic.com/tv/batwoman/user-reviews/?filter=Negative%20Reviews",
  "https://www.metacritic.com/tv/bridgerton/user-reviews/?filter=Positive%20Reviews",
  "https://www.metacritic.com/tv/batwoman/user-reviews/?filter=Positive%20Reviews",
  "https://www.metacritic.com/tv/wednesday/user-reviews/?filter=Positive%20Reviews",
  "https://www.metacritic.com/tv/alien-earth/user-reviews/?filter=Positive%20Reviews",
  "https://www.metacritic.com/tv/foundation/user-reviews/?filter=Positive%20Reviews",
  "https://www.metacritic.com/tv/chernobyl/user-reviews/?filter=Positive%20Reviews",
  "https://www.metacritic.com/tv/bojack-horseman/user-reviews/?filter=Positive%20Reviews"
)

allReviews <- data.frame(
  reviews = character(),
  sentiment = character(),
  stringsAsFactors = FALSE
)

for (url in urls) {
  webPage <- read_html(url)
  
  comment <- webPage %>%
    html_elements("div.c-siteReview_quote.g-outer-spacing-bottom-small > span") %>%
    html_text2()
  
  
  sentiment_label <- ifelse(str_detect(url, "Positive"), "positive", "negative")
  
  
  batch <- data.frame(
    reviews = comment,
    sentiment = sentiment_label,
    stringsAsFactors = FALSE
  )
  
  
  allReviews <- bind_rows(allReviews, batch)
  
  cat("Scraped", length(comment), sentiment_label, "reviews from", url, "\n")
}


table(allReviews$sentiment)


write.csv(allReviews, "reviews.csv", row.names = FALSE)














 



# 2. Preprocessing
csv_path <- "reviews.csv"

TEXT_COL <- "reviews"

raw <- read_csv(csv_path, locale = locale(encoding = "UTF-8"), show_col_types = FALSE)


if (!TEXT_COL %in% names(raw)) {
  candidates <- names(raw)[map_lgl(raw, ~ is.character(.x) || is.factor(.x))]
  TEXT_COL <- candidates[which.max(sapply(raw[candidates],
                                          function(x) mean(nchar(as.character(x)), na.rm = TRUE)))]
  message("Guessed text column: ", TEXT_COL)
}

df <- raw %>%
  mutate(doc_id = row_number(),
         text   = as.character(.data[[TEXT_COL]])) %>%
  select(doc_id, everything())


df$text <- iconv(df$text, from = "", to = "UTF-8")


cat("Rows:", nrow(df), "\n")

cat("NAs in text:", sum(is.na(df$text)), "\n")


df <- df %>%
  filter(!is.na(text)) %>%
  mutate(text = str_squish(text)) %>%
  filter(text != "") %>%
  distinct(text, .keep_all = TRUE)


df %>% select(doc_id, !!TEXT_COL := text) %>% head(5)

clean_text <- function(x) {
  x %>%
    str_replace_all("https?://\\S+|www\\.[^\\s]+", " ") %>%
    str_replace_all("@\\w+|#\\w+", " ") %>%
    str_replace_all("[^\\p{L}\\p{N}\\s']", " ") %>%
    str_replace_all("\\d+", " ") %>%                     
    str_replace_all("(.)\\1{2,}", "\\1\\1") %>%       
    str_to_lower() %>%
    str_squish()
}

df <- df %>% mutate(text_clean = clean_text(text))


df %>% select(doc_id, text, text_clean) %>% head(5)

data(stop_words)


custom_sw <- tibble(word = c("season","episode","episodes","series","show","shows","tv","netflix","hbo"),
                    lexicon = "custom")
stop_words <- bind_rows(stop_words, custom_sw)

# 3. Feature Extraction
tokens <- df %>%
  unnest_tokens(word, text_clean, token = "words") %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) >= 3)   

tokens %>% head(10)

tokens <- tokens %>%
  mutate(stem = SnowballC::wordStem(word, language = "en"))

tokens %>% select(doc_id, word, stem) %>% head(10)

word_freq <- tokens %>%
  count(stem, sort = TRUE)

head(word_freq, 20)

# 4. Visualization & Insights
top_n <- 20
p <- word_freq %>%
  slice_max(n, n = top_n) %>%
  ggplot(aes(x = reorder(stem, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = paste("Top", top_n, "words (by frequency)"),
       x = "Word (stem)", y = "Count") +
  theme_minimal()
p


ggsave("top_words_bar.png", p, width = 8, height = 5, dpi = 150)


par(mar = c(1,1,1,1))  
wc_df <- word_freq %>% filter(n > 1)

suppressWarnings(wordcloud(words = wc_df$stem, freq = wc_df$n, max.words = 200, random.order = FALSE))

dev.copy(png, filename = "wordcloud.png", width = 1000, height = 800)
dev.off()


tfidf <- tokens %>%
  count(doc_id, stem, sort = FALSE) %>%
  bind_tf_idf(term = stem, document = doc_id, n = n) %>%
  arrange(desc(tf_idf))

head(tfidf, 15)


tfidf_joined <- tfidf %>% left_join(df %>% select(doc_id, sentiment), by = "doc_id")
g_tfidf_box <- ggplot(tfidf_joined, aes(x = sentiment, y = tf_idf, fill = sentiment)) +
  geom_boxplot(show.legend = FALSE, outlier.shape = NA) +
  coord_cartesian(ylim = quantile(tfidf_joined$tf_idf, c(0.01, 0.99), na.rm = TRUE)) +
  labs(title = "TF-IDF distribution by class", x = "Sentiment", y = "TF-IDF") +
  theme_minimal()
print(g_tfidf_box)
ggsave("tfidf_box_by_class.png", g_tfidf_box, width = 6, height = 4, dpi = 150)




class_dist <- df %>% count(sentiment)
g_class <- ggplot(class_dist, aes(x = sentiment, y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Class distribution", x = "Sentiment", y = "Documents") +
  theme_minimal()
print(g_class)
ggsave("class_distribution.png", g_class, width = 6, height = 4, dpi = 150)


df_len <- df %>% mutate(doc_len = str_count(text_clean, "\\S+"))
g_len <- ggplot(df_len, aes(x = doc_len, fill = sentiment)) +
  geom_histogram(position = "identity", bins = 40, alpha = 0.6) +
  labs(title = "Document length distribution", x = "Tokens per document", y = "Count") +
  theme_minimal()
print(g_len)
ggsave("doc_length_hist.png", g_len, width = 7, height = 4, dpi = 150)


top_k <- 15
top_per_class <- tokens %>%
  count(sentiment, stem, sort = TRUE) %>%
  group_by(sentiment) %>%
  slice_max(n, n = top_k) %>%
  ungroup()
g_top_per_class <- ggplot(top_per_class, aes(x = reorder_within(stem, n, sentiment), y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_x_reordered() +
  facet_wrap(~ sentiment, scales = "free_y") +
  labs(title = paste("Top", top_k, "words per class (by frequency)"), x = "Word (stem)", y = "Count") +
  theme_minimal()
print(g_top_per_class)
ggsave("top_words_per_class.png", g_top_per_class, width = 9, height = 6, dpi = 150)



class_token_counts <- tokens %>% count(sentiment, stem, name = "n_wc")
totals_by_class <- class_token_counts %>% group_by(sentiment) %>% summarise(n_c = sum(n_wc), .groups = "drop")
V <- n_distinct(class_token_counts$stem)
likelihoods <- class_token_counts %>%
  left_join(totals_by_class, by = "sentiment") %>%
  mutate(p_wc = (n_wc + 1) / (n_c + V)) %>%
  select(sentiment, stem, p_wc) %>%
  pivot_wider(names_from = sentiment, values_from = p_wc, values_fill = 1 / (first(totals_by_class$n_c) + V))


if (all(c("negative", "positive") %in% names(likelihoods))) {
  log_odds <- likelihoods %>%
    mutate(log_odds = log((positive + 1e-12) / (negative + 1e-12))) %>%
    select(stem, log_odds)
  top_log_odds <- bind_rows(
    log_odds %>% slice_max(log_odds, n = top_k) %>% mutate(direction = "positive"),
    log_odds %>% slice_min(log_odds, n = top_k) %>% mutate(direction = "negative")
  )
  g_log_odds <- ggplot(top_log_odds, aes(x = reorder(stem, log_odds), y = log_odds, fill = direction)) +
    geom_col(show.legend = FALSE) +
    coord_flip() +
    labs(title = "Top words by Naive Bayes log-odds", x = "Word (stem)", y = "log(P(w|positive)/P(w|negative))") +
    theme_minimal()
  print(g_log_odds)
  ggsave("nb_log_odds_top_words.png", g_log_odds, width = 8, height = 6, dpi = 150)
}


tfidf <- tokens %>%
  count(doc_id, stem, sort = FALSE) %>%
  bind_tf_idf(term = stem, document = doc_id, n = n) %>%
  arrange(desc(tf_idf))

head(tfidf, 15)


dtm_tfidf <- tfidf %>%
  cast_dtm(document = doc_id, term = stem, value = tf_idf)

dtm_tfidf

dtm_counts <- tokens %>% count(doc_id, stem) %>% cast_dtm(doc_id, stem, n)

k <- 3
lda_model <- LDA(dtm_counts, k = k, control = list(seed = 1234))

 
topic_terms <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup()

topic_terms

write_csv(df,        "cleaned_text.csv")
write_csv(word_freq, "top_words.csv")
write_csv(tfidf,     "tfidf_by_doc.csv")
cat("Saved files: cleaned_text.csv, top_words.csv, tfidf_by_doc.csv, top_words_bar.png, wordcloud.png\n")















# 5. Sentiment Analysis


df <- read_csv("cleaned_text.csv")

tfidf_data <- read_csv("tfidf_by_doc.csv")



model_data_raw <- tfidf_data %>%
  left_join(df %>% select(doc_id, sentiment), by = "doc_id") %>%
  select(doc_id, stem, n, tf_idf, sentiment) %>%
  mutate(
    n = as.numeric(n),
    tf_idf = as.numeric(tf_idf)
  ) %>%
  filter(!is.na(sentiment))

df_by_stem <- model_data_raw %>%
  group_by(stem) %>%
  summarise(
    doc_freq = n_distinct(doc_id[n > 0], na.rm = TRUE),
    total_docs = n_distinct(doc_id),
    .groups = "drop"
  ) %>%
  mutate(df_prop = doc_freq / total_docs)

min_df <- 2            
max_df_prop <- 0.9     

valid_stems <- df_by_stem %>%
  filter(doc_freq >= min_df, df_prop < max_df_prop) %>%
  pull(stem)


if (length(valid_stems) < 50) {
  min_df <- 1
  max_df_prop <- 0.99
  valid_stems <- df_by_stem %>%
    filter(doc_freq >= min_df, df_prop < max_df_prop) %>%
    pull(stem)
}

model_data <- model_data_raw %>%
  filter(stem %in% valid_stems) %>%
  filter(str_detect(stem, "^[a-z]+$")) %>%
  filter(nchar(stem) >= 3, nchar(stem) <= 25)


if (nrow(model_data) == 0 || n_distinct(model_data$stem) == 0) {
  message("Fallback: using top stems by frequency without strict DF filters")
  top_stems <- model_data_raw %>%
    filter(str_detect(stem, "^[a-z]+$")) %>%
    filter(nchar(stem) >= 3, nchar(stem) <= 25) %>%
    group_by(stem) %>% summarise(total_n = sum(n, na.rm = TRUE), .groups = "drop") %>%
    slice_max(total_n, n = 500) %>% pull(stem)
  model_data <- model_data_raw %>% filter(stem %in% top_stems)
}

wide_data <- model_data %>%
  group_by(doc_id, sentiment, stem) %>%
  summarise(present = as.numeric(any(n > 0, na.rm = TRUE)), .groups = "drop") %>%
  pivot_wider(
    id_cols = c(doc_id, sentiment),
    names_from = stem,
    values_from = present,
    values_fill = list(present = 0),
    values_fn = list(present = max),
    names_prefix = "stem_"
  )


nzv_cols <- which(vapply(wide_data %>% select(-doc_id, -sentiment), function(col) length(unique(col)) > 1, logical(1)))
if (length(nzv_cols) > 0) {
  feature_names <- names((wide_data %>% select(-doc_id, -sentiment))[nzv_cols])
  wide_data <- bind_cols(
    wide_data %>% select(doc_id, sentiment),
    (wide_data %>% select(-doc_id, -sentiment))[, feature_names, drop = FALSE]
  )
}


wide_data$sentiment <- factor(wide_data$sentiment)

set.seed(123)
train_index <- createDataPartition(wide_data$sentiment, p = 0.8, list = FALSE)
train_data <- wide_data[train_index, ]
test_data <- wide_data[-train_index, ]

set.seed(123)
train_balanced <- downSample(x = train_data %>% select(-sentiment, -doc_id),
                             y = train_data$sentiment,
                             yname = "sentiment")

nb_model <- naiveBayes(sentiment ~ ., data = train_balanced, laplace = 1)

predictions <- predict(nb_model, test_data %>% select(-sentiment, -doc_id))


test_data$sentiment <- factor(test_data$sentiment)


predictions <- factor(predictions, levels = levels(test_data$sentiment))


confusion_matrix <- confusionMatrix(predictions, test_data$sentiment)
accuracy <- confusion_matrix$overall["Accuracy"]


cat("Model Accuracy:", round(accuracy * 100, 2), "%\n")
print(confusion_matrix)


feature_freq <- model_data %>%
  group_by(sentiment, stem) %>%
  summarise(doc_count = n_distinct(doc_id), .groups = 'drop') %>%
  group_by(sentiment) %>%
  slice_max(doc_count, n = 10) %>%
  arrange(sentiment, desc(doc_count))

cat("\nTop words for each sentiment (by document presence):\n")
print(feature_freq)


saveRDS(nb_model, "naive_bayes_model.rds")
write_csv(tibble(feature = names(train_balanced %>% select(-sentiment))), "model_features.csv")
