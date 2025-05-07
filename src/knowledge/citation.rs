use std::fmt;

/// Represents a bibliographic citation for a knowledge source
#[derive(Debug, Clone)]
pub struct Citation {
    /// The type of citation (e.g., "book", "article", "website")
    pub citation_type: String,
    /// Author of the cited work
    pub author: Option<String>,
    /// Title of the cited work
    pub title: Option<String>,
    /// Publication venue (journal, conference, publisher)
    pub publication: Option<String>,
    /// URL for online resources
    pub url: Option<String>,
    /// Publication date
    pub date: Option<String>,
    /// Page numbers for print resources
    pub page_numbers: Option<String>,
}

impl Citation {
    /// Create a new citation
    pub fn new(
        citation_type: &str,
        author: Option<&str>,
        title: Option<&str>,
        publication: Option<&str>,
        url: Option<&str>,
        date: Option<&str>,
        page_numbers: Option<&str>,
    ) -> Self {
        Self {
            citation_type: citation_type.to_string(),
            author: author.map(String::from),
            title: title.map(String::from),
            publication: publication.map(String::from),
            url: url.map(String::from),
            date: date.map(String::from),
            page_numbers: page_numbers.map(String::from),
        }
    }
    
    /// Format the citation in APA style
    pub fn format_apa(&self) -> String {
        match self.citation_type.as_str() {
            "book" => self.format_book_apa(),
            "article" => self.format_article_apa(),
            "website" => self.format_website_apa(),
            _ => self.format_generic_apa(),
        }
    }
    
    /// Format the citation in MLA style
    pub fn format_mla(&self) -> String {
        match self.citation_type.as_str() {
            "book" => self.format_book_mla(),
            "article" => self.format_article_mla(),
            "website" => self.format_website_mla(),
            _ => self.format_generic_mla(),
        }
    }
    
    /// Format a book citation in APA style
    fn format_book_apa(&self) -> String {
        let mut citation = String::new();
        
        // Author
        if let Some(author) = &self.author {
            citation.push_str(author);
            citation.push_str(". ");
        }
        
        // Date
        if let Some(date) = &self.date {
            citation.push_str("(");
            citation.push_str(date);
            citation.push_str("). ");
        }
        
        // Title (italicized)
        if let Some(title) = &self.title {
            citation.push_str("<em>");
            citation.push_str(title);
            citation.push_str("</em>. ");
        }
        
        // Publisher
        if let Some(publication) = &self.publication {
            citation.push_str(publication);
            citation.push_str(".");
        }
        
        citation
    }
    
    /// Format an article citation in APA style
    fn format_article_apa(&self) -> String {
        let mut citation = String::new();
        
        // Author
        if let Some(author) = &self.author {
            citation.push_str(author);
            citation.push_str(". ");
        }
        
        // Date
        if let Some(date) = &self.date {
            citation.push_str("(");
            citation.push_str(date);
            citation.push_str("). ");
        }
        
        // Title
        if let Some(title) = &self.title {
            citation.push_str(title);
            citation.push_str(". ");
        }
        
        // Journal name (italicized)
        if let Some(publication) = &self.publication {
            citation.push_str("<em>");
            citation.push_str(publication);
            citation.push_str("</em>");
        }
        
        // Page numbers
        if let Some(pages) = &self.page_numbers {
            citation.push_str(", ");
            citation.push_str(pages);
        }
        
        citation.push_str(".");
        
        citation
    }
    
    /// Format a website citation in APA style
    fn format_website_apa(&self) -> String {
        let mut citation = String::new();
        
        // Author
        if let Some(author) = &self.author {
            citation.push_str(author);
            citation.push_str(". ");
        }
        
        // Date
        if let Some(date) = &self.date {
            citation.push_str("(");
            citation.push_str(date);
            citation.push_str("). ");
        }
        
        // Title
        if let Some(title) = &self.title {
            citation.push_str(title);
            citation.push_str(". ");
        }
        
        // Website name
        if let Some(publication) = &self.publication {
            citation.push_str("<em>");
            citation.push_str(publication);
            citation.push_str("</em>. ");
        }
        
        // URL
        if let Some(url) = &self.url {
            citation.push_str("Retrieved from ");
            citation.push_str(url);
        }
        
        citation
    }
    
    /// Format a generic citation in APA style
    fn format_generic_apa(&self) -> String {
        let mut citation = String::new();
        
        // Author
        if let Some(author) = &self.author {
            citation.push_str(author);
            citation.push_str(". ");
        }
        
        // Date
        if let Some(date) = &self.date {
            citation.push_str("(");
            citation.push_str(date);
            citation.push_str("). ");
        }
        
        // Title
        if let Some(title) = &self.title {
            citation.push_str("<em>");
            citation.push_str(title);
            citation.push_str("</em>. ");
        }
        
        // Publication
        if let Some(publication) = &self.publication {
            citation.push_str(publication);
            citation.push_str(". ");
        }
        
        // URL
        if let Some(url) = &self.url {
            citation.push_str("Retrieved from ");
            citation.push_str(url);
        }
        
        citation
    }
    
    // MLA formatting methods
    
    /// Format a book citation in MLA style
    fn format_book_mla(&self) -> String {
        let mut citation = String::new();
        
        // Author
        if let Some(author) = &self.author {
            citation.push_str(author);
            citation.push_str(". ");
        }
        
        // Title (italicized)
        if let Some(title) = &self.title {
            citation.push_str("<em>");
            citation.push_str(title);
            citation.push_str("</em>. ");
        }
        
        // Publisher
        if let Some(publication) = &self.publication {
            citation.push_str(publication);
            citation.push_str(", ");
        }
        
        // Date
        if let Some(date) = &self.date {
            citation.push_str(date);
            citation.push_str(".");
        }
        
        citation
    }
    
    /// Format an article citation in MLA style
    fn format_article_mla(&self) -> String {
        let mut citation = String::new();
        
        // Author
        if let Some(author) = &self.author {
            citation.push_str(author);
            citation.push_str(". ");
        }
        
        // Title in quotes
        if let Some(title) = &self.title {
            citation.push_str("\"");
            citation.push_str(title);
            citation.push_str(".\" ");
        }
        
        // Journal name (italicized)
        if let Some(publication) = &self.publication {
            citation.push_str("<em>");
            citation.push_str(publication);
            citation.push_str("</em>");
        }
        
        // Date
        if let Some(date) = &self.date {
            citation.push_str(", ");
            citation.push_str(date);
        }
        
        // Page numbers
        if let Some(pages) = &self.page_numbers {
            citation.push_str(", pp. ");
            citation.push_str(pages);
        }
        
        citation.push_str(".");
        
        citation
    }
    
    /// Format a website citation in MLA style
    fn format_website_mla(&self) -> String {
        let mut citation = String::new();
        
        // Author
        if let Some(author) = &self.author {
            citation.push_str(author);
            citation.push_str(". ");
        }
        
        // Title in quotes
        if let Some(title) = &self.title {
            citation.push_str("\"");
            citation.push_str(title);
            citation.push_str(".\" ");
        }
        
        // Website name (italicized)
        if let Some(publication) = &self.publication {
            citation.push_str("<em>");
            citation.push_str(publication);
            citation.push_str("</em>");
        }
        
        // Date
        if let Some(date) = &self.date {
            citation.push_str(", ");
            citation.push_str(date);
        }
        
        // URL
        if let Some(url) = &self.url {
            citation.push_str(", ");
            citation.push_str(url);
        }
        
        citation.push_str(".");
        
        citation
    }
    
    /// Format a generic citation in MLA style
    fn format_generic_mla(&self) -> String {
        let mut citation = String::new();
        
        // Author
        if let Some(author) = &self.author {
            citation.push_str(author);
            citation.push_str(". ");
        }
        
        // Title (italicized)
        if let Some(title) = &self.title {
            citation.push_str("<em>");
            citation.push_str(title);
            citation.push_str("</em>. ");
        }
        
        // Publication info
        if let Some(publication) = &self.publication {
            citation.push_str(publication);
            citation.push_str(", ");
        }
        
        // Date
        if let Some(date) = &self.date {
            citation.push_str(date);
        }
        
        // URL
        if let Some(url) = &self.url {
            citation.push_str(", ");
            citation.push_str(url);
        }
        
        citation.push_str(".");
        
        citation
    }
}

impl fmt::Display for Citation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_apa())
    }
}

/// CitationManager handles collections of citations and bibliography generation
pub struct CitationManager {
    /// List of all citations
    citations: Vec<Citation>,
}

impl CitationManager {
    /// Create a new citation manager
    pub fn new() -> Self {
        Self {
            citations: Vec::new(),
        }
    }
    
    /// Add a citation
    pub fn add_citation(&mut self, citation: Citation) {
        self.citations.push(citation);
    }
    
    /// Get all citations
    pub fn citations(&self) -> &[Citation] {
        &self.citations
    }
    
    /// Generate a bibliography in APA style
    pub fn generate_apa_bibliography(&self) -> String {
        let mut bibliography = String::new();
        
        for citation in &self.citations {
            bibliography.push_str(&citation.format_apa());
            bibliography.push_str("\n\n");
        }
        
        bibliography
    }
    
    /// Generate a bibliography in MLA style
    pub fn generate_mla_bibliography(&self) -> String {
        let mut bibliography = String::new();
        
        for citation in &self.citations {
            bibliography.push_str(&citation.format_mla());
            bibliography.push_str("\n\n");
        }
        
        bibliography
    }
    
    /// Clear all citations
    pub fn clear(&mut self) {
        self.citations.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_book_citation_apa() {
        let citation = Citation::new(
            "book",
            Some("Smith, J."),
            Some("The Art of Programming"),
            Some("Tech Press"),
            None,
            Some("2023"),
            None,
        );
        
        let formatted = citation.format_apa();
        assert!(formatted.contains("Smith, J."));
        assert!(formatted.contains("(2023)"));
        assert!(formatted.contains("<em>The Art of Programming</em>"));
        assert!(formatted.contains("Tech Press"));
    }
    
    #[test]
    fn test_website_citation_mla() {
        let citation = Citation::new(
            "website",
            Some("Johnson, A."),
            Some("Modern Web Development"),
            Some("Dev Blog"),
            Some("https://example.com/blog"),
            Some("2023"),
            None,
        );
        
        let formatted = citation.format_mla();
        assert!(formatted.contains("Johnson, A."));
        assert!(formatted.contains("\"Modern Web Development.\""));
        assert!(formatted.contains("<em>Dev Blog</em>"));
        assert!(formatted.contains("https://example.com/blog"));
    }
    
    #[test]
    fn test_citation_manager() {
        let mut manager = CitationManager::new();
        
        let citation1 = Citation::new(
            "book",
            Some("Smith, J."),
            Some("The Art of Programming"),
            Some("Tech Press"),
            None,
            Some("2023"),
            None,
        );
        
        let citation2 = Citation::new(
            "article",
            Some("Johnson, K."),
            Some("Modern Programming Paradigms"),
            Some("Journal of Computer Science"),
            None,
            Some("2022"),
            Some("45-67"),
        );
        
        manager.add_citation(citation1);
        manager.add_citation(citation2);
        
        let apa_bibliography = manager.generate_apa_bibliography();
        assert!(apa_bibliography.contains("Smith, J."));
        assert!(apa_bibliography.contains("Johnson, K."));
        
        let mla_bibliography = manager.generate_mla_bibliography();
        assert!(mla_bibliography.contains("Smith, J."));
        assert!(mla_bibliography.contains("Journal of Computer Science"));
    }
} 