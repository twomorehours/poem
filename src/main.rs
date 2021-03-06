// lazy_static 实际上是定义了一个新类型 这个类型里面保存指定的类型
// 然后实现Deref<指定类型> 并且在第一次deref的时候实例化指定类型 并保存在static的新类型值中
use anyhow::Result;
use cang_jie::{CangJieTokenizer, TokenizerOption, CANG_JIE};
use clap::{AppSettings, Parser};
use colored::*;
use indicatif::ProgressBar;
use jieba_rs::Jieba;
use lazy_static::{__Deref, lazy_static};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{atomic::AtomicUsize, Arc},
};
use tantivy::{
    collector::TopDocs,
    doc,
    query::QueryParser,
    schema::{Field, IndexRecordOption, Schema, SchemaBuilder, TextFieldIndexing, TextOptions},
    Document, Index,
};

const POEMS_STR: &str = include_str!("../poems.json");

#[derive(Parser, Debug)]
#[clap(about, version, author)]
#[clap(global_setting(AppSettings::AllowNegativeNumbers))]
struct Args {
    #[clap(subcommand)]
    action: Action,
}

#[derive(clap::Subcommand, Debug)]
enum Action {
    /// index all poems
    Index {
        /// the path index will be stored
        #[clap(long, parse(from_os_str), default_value = ".poem_index")]
        index_path: PathBuf,
    },

    /// search poems
    Search {
        /// the path index is stored
        #[clap(long, parse(from_os_str), default_value = ".poem_index")]
        index_path: PathBuf,
        /// the keyword
        keyword: String,
    },

    /// list poems
    List {
        /// the max count of poem list
        #[clap(long)]
        limit: Option<usize>,
    },
    /// get random poems
    Random {
        /// the count you need
        #[clap(long, default_value = "1")]
        count: usize,
    },

    /// get stat of all poems
    Stat {
        /// sort by count desc
        #[clap(long)]
        sort: bool,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.action {
        Action::Index { index_path } => {
            let index = open_or_create_index(index_path, false)?;
            let mut writer = index.writer(1024 * 1024 * 10)?;
            let poems: Vec<Poem> = serde_json::from_str(POEMS_STR)?;
            let bar = ProgressBar::new(poems.len() as _);
            poems.into_iter().map(Document::from).for_each(|doc| {
                writer.add_document(doc);
                bar.inc(1);
            });
            writer.commit()?;
            bar.finish();
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
            let top_docs = searcher.search(query.as_ref(), &TopDocs::with_limit(10000))?;
            for (_, doc_address) in top_docs.into_iter() {
                let poem: Poem = searcher.doc(doc_address)?.into();
                println!("{}", poem);
            }
        }
        Action::List { limit } => {
            let poems: Vec<Poem> = serde_json::from_str(POEMS_STR)?;
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
        Action::Random { mut count } => {
            let poems: Vec<Poem> = serde_json::from_str(POEMS_STR)?;
            if poems.is_empty() {
                println!("no poem in repo");
                return Ok(());
            }
            if count > poems.len() {
                count = poems.len();
            }
            let mut rng = rand::thread_rng();
            let mut set = HashSet::new();
            while set.len() < count {
                set.insert(&poems[rng.gen_range(0..poems.len())]);
            }
            set.into_iter().for_each(|p| println!("{}", p));
        }
        Action::Stat { sort } => {
            let poems: Vec<Poem> = serde_json::from_str(POEMS_STR)?;
            let dynasty: Vec<&str> = poems.iter().map(|p| &p.dynasty[..]).collect();
            let author: Vec<&str> = poems.iter().map(|p| &p.author[..]).collect();

            let stat = Stat::new(
                poems.len() as _,
                words_count(&author, sort),
                words_count(&dynasty, sort),
            );
            println!("{}", stat);
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

#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, Clone)]
struct Poem {
    title: String,
    author: String,
    dynasty: String,
    content: String,
}

impl Display for Poem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\t{}", self.title.bright_cyan()).unwrap();
        writeln!(f, "\t{}〔{}〕", self.author.cyan(), self.dynasty.cyan()).unwrap();
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

#[derive(Debug)]
struct Stat<'a> {
    total: i32,
    author: Vec<(&'a str, i32)>,
    dynasty: Vec<(&'a str, i32)>,
}

impl<'a> Stat<'a> {
    fn new(total: i32, author: Vec<(&'a str, i32)>, dynasty: Vec<(&'a str, i32)>) -> Self {
        Self {
            total,
            author,
            dynasty,
        }
    }
}

impl<'a> Display for Stat<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "总数：{}", self.total).unwrap();
        writeln!(f, "朝代：").unwrap();
        self.dynasty
            .iter()
            .for_each(|d| writeln!(f, "{:>7}：{}", d.0, d.1).unwrap());
        writeln!(f, "作者：").unwrap();
        self.author
            .iter()
            .for_each(|d| writeln!(f, "{:>7}： {}", d.0, d.1).unwrap());
        Ok(())
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

fn words_count<'a>(words: &[&'a str], sort: bool) -> Vec<(&'a str, i32)> {
    let mut map = HashMap::new();
    for w in words {
        let entry = map.entry(*w).or_insert(0);
        *entry += 1;
    }
    let mut pairs = map.into_iter().collect::<Vec<_>>();
    if sort {
        pairs.sort_by_key(|p| -p.1);
    }
    pairs
}
