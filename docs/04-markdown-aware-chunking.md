# Markdown-Aware Chunking: Preserving Document Structure

Most technical documentation, README files, and knowledge base articles are written in markdown. Markdown encodes structure — headings, tables, code blocks, lists, block quotes — using lightweight syntax that's designed for humans to read but also parseable by machines. When a chunker treats markdown as plain text, that structure is invisible. The results can be surprisingly bad.

## What goes wrong with plain text chunking

Consider a markdown table:

```markdown
| Model          | Parameters | Accuracy |
|----------------|-----------|----------|
| BERT-base      | 110M      | 88.5%    |
| RoBERTa-large  | 355M      | 90.2%    |
| DeBERTa-v3     | 304M      | 91.1%    |
```

A fixed-size chunker with a 200-byte target might split this between the header row and the data rows. The first chunk contains column names without any values. The second chunk contains numbers without any column headers to explain what they mean. Neither chunk makes sense on its own, and an embedding model can't recover the meaning from either fragment.

The same problem afflicts code blocks. A function definition split across two chunks produces one chunk with a signature and no body, and another with a body and no signature. Neither is useful for retrieval. If someone searches for "how does the authentication function work," a chunk containing only the function body without its name won't match well.

Lists suffer similarly. A bullet point "- Use `--threshold 0.5` for moderate sensitivity" only makes sense in the context of the list it belongs to. Split from its siblings, it's an orphaned instruction with no context about what threshold it's referring to or what the alternatives are.

## AST-based block extraction

The solution is to parse the markdown into an abstract syntax tree before chunking. An AST parser understands the nesting structure of the document: which lines belong to a table, where a code block starts and ends, which items form a list. This structural understanding lets the chunker keep compound elements intact.

The parsing uses a streaming approach that walks the AST events and tracks nesting depth. When it encounters the start of a table, code block, list, or block quote, it records the position and begins accumulating content. When it encounters the matching end event at the same nesting depth, it emits the entire element as a single atomic block. No matter how large the table is or how many lines the code block spans, it stays together as one unit.

Paragraphs receive different treatment. A paragraph is typically not a single atomic unit — it might contain several sentences about different aspects of the same topic. The parser extracts the full paragraph text and then applies Unicode sentence segmentation to split it into individual sentences. Each sentence becomes its own block, giving the semantic chunking pipeline fine-grained boundaries to work with within continuous prose.

Headings become their own blocks as well. This is important because headings are high-signal text: they summarize the topic of the following section and strongly influence embedding similarity. Keeping them separate means the semantic pipeline can detect the boundary between sections by observing the similarity shift at the heading.

## The block types

The parser produces six kinds of blocks:

**Sentence** blocks come from paragraphs that have been sentence-split. They're the most numerous block type in typical documents and provide the granularity that lets the semantic pipeline detect topic shifts within prose.

**Table** blocks contain the complete markdown table — header row, separator row, and all data rows. The entire table is embedded as a single unit, which means its embedding captures the relationship between columns and values rather than treating each row as an independent statement.

**Code block** elements preserve the complete fenced or indented code block, including the language tag. A function definition, a configuration example, or a shell command sequence stays together as one block.

**Heading** blocks contain the heading text including its level markers. They serve as topic anchors in the embedding space — the semantic pipeline uses their position to identify section boundaries.

**List** blocks contain the complete list with all its items. Ordered and unordered lists are both kept atomic, as are task lists. The embedding of a full list captures the relationship between items, which is often more useful for retrieval than any single item's embedding.

**Block quote** elements preserve the quoted text as a single unit, maintaining the distinction between quoted and original content.

## How this integrates with the semantic pipeline

The markdown-aware block extraction is the first stage of the semantic chunking pipeline described in the previous article. Instead of sentence-splitting the entire document as plain text, the parser produces a mixed sequence of block types: sentences interleaved with tables, code blocks, headings, and lists.

All of these blocks — regardless of type — get embedded in the same vector space. The semantic pipeline then computes cross-similarity between adjacent blocks and looks for topic boundaries exactly as described before. The key difference is that structural elements participate in the similarity computation as whole units. A table about performance metrics will have a different embedding than the surrounding prose about implementation details, creating a natural similarity dip that the pipeline can detect.

This means the semantic chunker automatically groups related content: introductory prose, its accompanying table, and the analysis paragraph that follows might all end up in the same chunk if they're about the same topic. But a code block about a different feature, even if it's only a few lines below, gets placed into a separate chunk because the embedding similarity drops at the boundary.

## Byte offset tracking

Every block carries its byte offset in the original document. This matters for two reasons.

First, it allows reconstructing the exact position of each chunk in the source document. When the pipeline groups blocks into chunks at topic boundaries, it can report the byte offset where each chunk begins. This is essential for applications that need to link retrieved chunks back to their source location — for example, highlighting the relevant passage in a document viewer.

Second, the offsets verify correctness. During testing, the parser checks that extracting the text at each block's reported offset from the original document reproduces the block text exactly. If the offsets are wrong, the test fails, catching any regression in the parsing logic.

## When markdown-awareness isn't needed

Not every document is markdown. For plain text files, email bodies, transcripts, and other unstructured content, the markdown parser would find no structural elements and produce only sentence blocks. In these cases, plain sentence splitting is faster because it skips the AST parsing step entirely.

The semantic chunking pipeline supports both modes: markdown-aware block extraction (the default) and plain text sentence splitting. Use the plain text mode when your input doesn't contain markdown syntax, or when the markdown structure isn't meaningful for your use case (for example, if you're chunking a markdown file where the tables and code blocks aren't important for retrieval).

## The practical impact

The difference markdown-awareness makes is most visible in documents that mix prose with structured elements. A technical specification with tables of parameters, a tutorial with code examples, a research paper with data tables — these documents contain high-value structured content that naive chunking destroys.

When a user asks "what are the default parameters for the semantic chunker?" and the answer lives in a markdown table, a markdown-aware chunker gives the retrieval system a chunk containing the complete table. A naive chunker might give it a chunk containing half the table and half of the surrounding paragraph. The first produces a correct, complete answer. The second produces a confusing fragment that the LLM has to work around — or fails on entirely.

For documentation and knowledge base applications, this difference is the reason to prefer markdown-aware chunking despite its slightly higher parsing cost.
