use anyhow::Result;
use cang_jie::{CangJieTokenizer, TokenizerOption, CANG_JIE};
use clap::Parser;
use colored::*;
use jieba_rs::Jieba;
use rand::prelude::SliceRandom;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::Display,
    fs,
    path::Path,
    sync::Arc,
    time::Instant,
};
use tantivy::{
    collector::TopDocs,
    doc,
    query::QueryParser,
    schema::{Field, IndexRecordOption, Schema, SchemaBuilder, TextFieldIndexing, TextOptions},
    Document, Index,
};

#[derive(Parser, Debug)]
#[clap(about = "A repo for poems", version = "1.0.0")]
struct Args {
    #[clap(subcommand)]
    action: Action,
}

#[derive(clap::Subcommand, Debug)]
enum Action {
    #[clap(about = "index all poems")]
    Index {
        /// the file path of poems
        #[clap(long)]
        src_path: String,

        /// the path index will be stored
        #[clap(long, default_value = "poem_index")]
        index_path: String,
    },

    #[clap(about = "search poems")]
    Search {
        /// the path index is stored
        #[clap(long, default_value = "poem_index")]
        index_path: String,
        /// the keyword
        #[clap(long)]
        keyword: String,
    },

    #[clap(about = "list poems")]
    List {
        /// the file path of poems
        #[clap(long)]
        src_path: String,
        /// the max count of poem list
        #[clap(long)]
        limit: Option<usize>,
    },
    #[clap(about = "get a random poem")]
    Random {
        /// the file path of poems
        #[clap(long)]
        src_path: String,
        /// the count you need
        #[clap(long, default_value = "1")]
        count: usize,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.action {
        Action::Index {
            src_path,
            index_path,
        } => {
            let poems_str = fs::read_to_string(&src_path)?;
            let poems: Vec<Poem> = serde_json::from_str(&poems_str)?;
            let len = poems.len();
            let index = open_or_create_index(index_path, false)?;
            let mut writer = index.writer(1024 * 1024 * 10)?;
            let start = Instant::now();
            poems.into_iter().map(Document::from).for_each(|doc| {
                writer.add_document(doc);
            });
            writer.commit()?;
            let elapsed = Instant::now().duration_since(start).as_millis();
            println!("indxed {} poems in {} ms", len, elapsed);
        }
        Action::Search {
            index_path,
            keyword,
        } => {
            let index = open_or_create_index(index_path, true)?;
            let reader = index.reader()?;
            let searcher = reader.searcher();
            let (_, fields) = build_schema();

            let query = QueryParser::for_index(&index, fields.into_values().into_iter().collect())
                .parse_query(&keyword)?;
            let start = Instant::now();
            let top_docs = searcher.search(query.as_ref(), &TopDocs::with_limit(10000))?;
            let len = top_docs.len();
            let elapsed = Instant::now().duration_since(start).as_millis();
            for (_, doc_address) in top_docs.into_iter() {
                let poem: Poem = searcher.doc(doc_address)?.into();
                println!("{}", poem);
            }
            println!("got {} poems in {} ms", len, elapsed);
        }
        Action::List { src_path, limit } => {
            let poems_str = fs::read_to_string(&src_path)?;
            let poems: Vec<Poem> = serde_json::from_str(&poems_str)?;
            let poems = match limit {
                Some(l) => {
                    if l > poems.len() {
                        &poems[..]
                    } else {
                        &poems[..l]
                    }
                }
                None => &poems[..],
            };
            poems.iter().for_each(|p| println!("{}", p));
        }
        Action::Random {
            src_path,
            mut count,
        } => {
            let poems_str = fs::read_to_string(&src_path)?;
            let mut poems: Vec<Poem> = serde_json::from_str(&poems_str)?;
            if poems.is_empty() {
                println!("no poem in repo");
                return Ok(());
            }
            if count > poems.len() {
                count = poems.len();
            }
            let mut rng = rand::thread_rng();
            poems.shuffle(&mut rng);
            poems.iter().take(count).for_each(|p| println!("{}", p));
        }
    }

    Ok(())
}

fn open_or_create_index(path: impl AsRef<Path>, read_only: bool) -> Result<Index> {
    let (schema, _) = build_schema();

    let path = path.as_ref();

    let index = if read_only {
        Index::open_in_dir(path)?
    } else {
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
        fs::create_dir_all(path)?;
        Index::create_in_dir(path, schema)?
    };
    index.tokenizers().register(CANG_JIE, tokenizer());

    Ok(index)
}

fn build_schema() -> (Schema, HashMap<&'static str, Field>) {
    let mut schema_builder = SchemaBuilder::default();

    let text_indexing = TextFieldIndexing::default()
        .set_tokenizer(CANG_JIE) // Set custom tokenizer
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let text_options = TextOptions::default()
        .set_indexing_options(text_indexing)
        .set_stored();

    let title = schema_builder.add_text_field("title", text_options.clone());
    let author = schema_builder.add_text_field("author", text_options.clone());
    let dynasty = schema_builder.add_text_field("dynasty", text_options.clone());
    let content = schema_builder.add_text_field("content", text_options);

    let schema = schema_builder.build();

    let mut fileds = HashMap::with_capacity(4);
    fileds.insert("title", title);
    fileds.insert("author", author);
    fileds.insert("dynasty", dynasty);
    fileds.insert("content", content);

    (schema, fileds)
}

fn tokenizer() -> CangJieTokenizer {
    CangJieTokenizer {
        worker: Arc::new(Jieba::empty()), // empty dictionary
        option: TokenizerOption::Unicode,
    }
}

#[derive(Debug, Serialize, Deserialize, Eq)]
struct Poem {
    title: String,
    author: String,
    dynasty: String,
    content: String,
}

impl Display for Poem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\t{}", self.title.cyan().bold()).unwrap();
        writeln!(f, "\t{}({})", self.author.cyan(), self.dynasty.cyan()).unwrap();
        writeln!(f, "{}", self.content.cyan())
    }
}

impl From<Poem> for Document {
    fn from(p: Poem) -> Self {
        let (_, fields) = build_schema();
        let mut doc = Document::new();
        doc.add_text(*fields.get("title").unwrap(), p.title);
        doc.add_text(*fields.get("author").unwrap(), p.author);
        doc.add_text(*fields.get("dynasty").unwrap(), p.dynasty);
        doc.add_text(*fields.get("content").unwrap(), p.content);
        doc
    }
}

impl From<Document> for Poem {
    fn from(doc: Document) -> Self {
        let (_, fields) = build_schema();
        Self {
            title: extract_field_text(&doc, *fields.get("title").unwrap()),
            author: extract_field_text(&doc, *fields.get("author").unwrap()),
            dynasty: extract_field_text(&doc, *fields.get("dynasty").unwrap()),
            content: extract_field_text(&doc, *fields.get("content").unwrap()),
        }
    }
}

impl PartialEq for Poem {
    fn eq(&self, other: &Self) -> bool {
        self.title == other.title && self.author == other.author
    }
}

fn extract_field_text(doc: &Document, field: Field) -> String {
    doc.get_all(field)
        .next()
        .unwrap()
        .text()
        .unwrap()
        .to_string()
}