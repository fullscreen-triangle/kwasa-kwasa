# Turbulance Language Syntax Specification

## Overview

Turbulance is a domain-specific language designed for semantic text processing and metacognitive analysis. This document provides the complete formal syntax specification for the language.

## Lexical Structure

### Tokens

#### Keywords
```
funxn       function declaration
var         variable declaration
let         immutable binding
given       conditional expression
within      scope boundary
proposition logical container
motion      logical unit within proposition
cause       causal relationship
considering iteration over collections
allow       permission granting
ensure      constraint enforcement
return      function return
import      module import
try         error handling
catch       exception handling
parallel    concurrent execution
project     text unit projection
support     motion support
oppose      motion opposition
if          conditional branching
else        alternative branch
while       loop construct
for         iteration construct
break       loop termination
continue    loop continuation
true        boolean literal
false       boolean literal
null        null value
```

#### Operators
```
Arithmetic:
/    division (text unit splitting)
*    multiplication (text unit combination)  
+    addition (text concatenation)
-    subtraction (text removal)
%    modulo (text sampling)
**   exponentiation (text amplification)

Comparison:
==   equality
!=   inequality
<    less than
<=   less than or equal
>    greater than
>=   greater than or equal

Logical:
&&   logical AND
||   logical OR
!    logical NOT

Assignment:
=    assignment
+=   addition assignment
-=   subtraction assignment
*=   multiplication assignment
/=   division assignment

Pipeline:
|>   pipeline operator
->   function type arrow
=>   lambda arrow

Scope:
::   scope resolution
.    member access
```

#### Literals
```
String:
"double quoted"
'single quoted'
`template literals with ${expressions}`

Numeric:
42          integer
3.14        float
1.23e10     scientific notation
0x1A2B      hexadecimal
0b1010      binary
0o755       octal

Regular Expression:
/pattern/flags
```

#### Identifiers
```
identifier ::= [a-zA-Z_][a-zA-Z0-9_]*

Valid examples:
variable_name
functionName
_private
camelCase
snake_case
```

#### Comments
```
// Single line comment
/* Multi-line
   comment */
/// Documentation comment
```

## Grammar Specification

### Program Structure
```ebnf
program         ::= item*
item            ::= function_def | import_stmt | proposition_def | type_def

function_def    ::= "funxn" identifier "(" parameter_list? ")" ":" type? block
parameter_list  ::= parameter ("," parameter)*
parameter       ::= identifier ":" type ("=" expression)?

import_stmt     ::= "import" module_path ("as" identifier)?
module_path     ::= identifier ("::" identifier)*
```

### Type System
```ebnf
type            ::= primitive_type | compound_type | function_type
primitive_type  ::= "TextUnit" | "String" | "i32" | "f64" | "bool" | "char"
compound_type   ::= array_type | tuple_type | custom_type
array_type      ::= "Array" "<" type ">"
tuple_type      ::= "(" type ("," type)* ")"
function_type   ::= "(" type_list? ")" "->" type
custom_type     ::= identifier type_args?
type_args       ::= "<" type ("," type)* ">"
```

### Expressions
```ebnf
expression      ::= assignment_expr
assignment_expr ::= logical_or_expr (assignment_op logical_or_expr)?
assignment_op   ::= "=" | "+=" | "-=" | "*=" | "/="

logical_or_expr ::= logical_and_expr ("||" logical_and_expr)*
logical_and_expr::= equality_expr ("&&" equality_expr)*
equality_expr   ::= relational_expr (("==" | "!=") relational_expr)*
relational_expr ::= additive_expr (("<" | "<=" | ">" | ">=") additive_expr)*

additive_expr   ::= multiplicative_expr (("+" | "-") multiplicative_expr)*
multiplicative_expr ::= unary_expr (("*" | "/" | "%") unary_expr)*
unary_expr      ::= ("!" | "-" | "+")? power_expr
power_expr      ::= postfix_expr ("**" unary_expr)?

postfix_expr    ::= primary_expr postfix_op*
postfix_op      ::= member_access | function_call | array_index
member_access   ::= "." identifier
function_call   ::= "(" argument_list? ")"
array_index     ::= "[" expression "]"

argument_list   ::= expression ("," expression)*
```

### Primary Expressions
```ebnf
primary_expr    ::= literal | identifier | parenthesized_expr | 
                   lambda_expr | text_operation | pipeline_expr

literal         ::= string_literal | numeric_literal | boolean_literal | 
                   regex_literal | null_literal

parenthesized_expr ::= "(" expression ")"
lambda_expr     ::= "|" parameter_list? "|" expression
pipeline_expr   ::= expression "|>" function_call
```

### Text Operations
```ebnf
text_operation  ::= division_op | multiplication_op | addition_op | subtraction_op

division_op     ::= expression "/" boundary_type
boundary_type   ::= "sentence" | "paragraph" | "word" | "character" | 
                   "section" | "phrase" | "line" | custom_boundary

multiplication_op ::= expression ("*" expression)+
addition_op     ::= expression "+" expression
subtraction_op  ::= expression "-" expression

custom_boundary ::= identifier | function_call
```

### Statements
```ebnf
statement       ::= expression_stmt | var_declaration | if_stmt | 
                   while_stmt | for_stmt | given_stmt | within_stmt |
                   return_stmt | break_stmt | continue_stmt | block

expression_stmt ::= expression ";"?
var_declaration ::= ("var" | "let") identifier (":" type)? ("=" expression)?

if_stmt         ::= "if" expression block ("else" (if_stmt | block))?
while_stmt      ::= "while" expression block
for_stmt        ::= "for" identifier "in" expression block

given_stmt      ::= "given" condition ":" block
condition       ::= expression
within_stmt     ::= "within" expression ":" block

return_stmt     ::= "return" expression?
break_stmt      ::= "break"
continue_stmt   ::= "continue"

block           ::= "{" statement* "}"
```

### Propositions and Motions
```ebnf
proposition_def ::= "proposition" identifier ":" proposition_body
proposition_body ::= "{" motion_def* proposition_logic* "}"

motion_def      ::= "motion" identifier "(" string_literal ")"
motion_logic    ::= "within" expression ":" logic_block
logic_block     ::= "{" logic_statement* "}"

logic_statement ::= given_support | given_oppose | motion_operation
given_support   ::= "given" condition ":" "support" identifier
given_oppose    ::= "given" condition ":" "oppose" identifier

motion_operation ::= identifier "." motion_method "(" argument_list? ")"
motion_method   ::= "spelling" | "capitalization" | "check_bias" | 
                   "find_evidence" | "logical_consistency"
```

### Specialized Constructs

#### Considering Statements
```ebnf
considering_stmt ::= "considering" collection_expr filter_clause? ":" block
collection_expr  ::= "all" type_name "in" expression | 
                    "these" expression |
                    expression

filter_clause    ::= "where" expression
type_name        ::= "paragraphs" | "sentences" | "words" | "motions" | 
                    "sections" | custom_type_name
```

#### Cause Declarations
```ebnf
cause_decl      ::= "cause" identifier "=" cause_body
cause_body      ::= "{" cause_field* "}"
cause_field     ::= field_name ":" expression ","?
field_name      ::= "primary" | "effects" | "confidence" | 
                   "temporal_relation" | "evidence"
```

#### Allow Statements
```ebnf
allow_stmt      ::= "allow" permission_list "on" target_expr
permission_list ::= permission ("," permission)*
permission      ::= identifier | function_call
target_expr     ::= identifier | member_access
```

#### Parallel Execution
```ebnf
parallel_expr   ::= "parallel" "{" expression_list "}"
expression_list ::= expression ("," expression)*
```

### Error Handling
```ebnf
try_expr        ::= "try" block catch_clause*
catch_clause    ::= "catch" exception_pattern block
exception_pattern ::= identifier | identifier "(" identifier ")"
```

## Semantic Rules

### Type Compatibility

#### Text Unit Hierarchy
```
TextUnit (base type)
├── Document
├── Section  
├── Paragraph
├── Sentence
├── Phrase
├── Word
└── Character
```

#### Operation Type Rules
```turbulance
// Division: converts to array of smaller units
Paragraph / sentence -> Array<Sentence>
Sentence / word -> Array<Word>

// Multiplication: combines units of same type
Sentence * Sentence -> Paragraph
Word * Word -> Phrase

// Addition: extends unit with compatible content
Sentence + String -> Sentence
Paragraph + Sentence -> Paragraph

// Subtraction: removes content from unit
TextUnit - String -> TextUnit
TextUnit - Pattern -> TextUnit
```

### Scope Rules

#### Variable Scoping
- Block scope for `var` declarations
- Function scope for parameters
- Module scope for top-level items
- Proposition scope for motions

#### Text Unit Scoping
```turbulance
within document:
    // 'document' is available as implicit variable
    var sections = document / section
    
    within section:
        // both 'document' and 'section' available
        var paragraphs = section / paragraph
```

### Motion and Proposition Semantics

#### Motion Types
```turbulance
motion claim = Motion("Statement to be verified", "claim")
motion evidence = Motion("Supporting information", "evidence") 
motion assumption = Motion("Assumed premise", "assumption")
motion conclusion = Motion("Logical conclusion", "conclusion")
```

#### Support/Oppose Semantics
```turbulance
proposition Example:
    motion Clarity("Text should be clear")
    
    within text:
        given readability_score() > 70:
            support Clarity    // Increases motion confidence
        given contains_jargon():
            oppose Clarity     // Decreases motion confidence
```

## Standard Library Integration

### Built-in Functions
```turbulance
// Text analysis
readability_score(text: TextUnit) -> f64
sentiment_analysis(text: TextUnit) -> Sentiment
extract_keywords(text: TextUnit, count: i32) -> Array<String>

// Text transformation  
simplify_sentences(text: TextUnit) -> TextUnit
formalize(text: TextUnit, level: String) -> TextUnit
replace_jargon(text: TextUnit, domain: String) -> TextUnit

// Research and validation
research_context(topic: String) -> ResearchData
fact_check(statement: String) -> FactCheckResult
ensure_explanation_follows(term: String) -> TextUnit

// Utilities
print(value: Any) -> Unit
len(collection: Array<T>) -> i32
typeof(value: Any) -> String
```

### Domain-Specific Extensions
```turbulance
// Genomic extension
import genomic
var dna = genomic.Sequence("ATGC...")
var codons = dna / codon

// Chemistry extension  
import chemistry
var molecule = chemistry.from_smiles("CC(=O)O")
var fragments = molecule / functional_group

// Mass spectrometry extension
import mass_spec
var spectrum = mass_spec.load("data.mzML")
var peaks = spectrum / peak
```

## Syntax Examples

### Complete Function Example
```turbulance
funxn enhance_academic_text(text: TextUnit, domain: String = "general") -> TextUnit:
    var enhanced = text
    
    // Check readability and improve if needed
    given readability_score(enhanced) < 60:
        enhanced = simplify_sentences(enhanced)
    
    // Replace domain-specific jargon
    enhanced = replace_jargon(enhanced, domain)
    
    // Ensure technical terms are explained
    considering all word in enhanced where is_technical_term(word):
        enhanced = ensure_explanation_follows(word.text, enhanced)
    
    // Add appropriate transitions
    enhanced = add_transitions(enhanced)
    
    return formalize(enhanced, "academic")

// Usage with proposition
proposition AcademicQuality:
    motion Clarity("Academic text should be clear to target audience")
    motion Rigor("Claims should be well-supported")
    motion Coherence("Ideas should flow logically")
    
    within enhanced_text:
        given readability_score() > 65:
            support Clarity
        given citation_density() > 0.1:
            support Rigor
        given transition_quality() > 0.7:
            support Coherence
```

### Complex Text Processing Pipeline
```turbulance
funxn process_research_paper(paper: Document) -> Document:
    var processed = paper
    
    // Analyze each section differently
    considering all section in processed:
        given section.type == "abstract":
            section = section 
                |> ensure_conciseness()
                |> highlight_key_findings()
                
        given section.type == "methodology":
            section = section
                |> ensure_reproducibility()
                |> add_technical_details()
                
        given section.type == "results":
            section = section
                |> validate_statistics()
                |> ensure_clear_presentation()
    
    // Cross-section analysis
    var coherence = check_cross_section_coherence(processed)
    given coherence < 0.8:
        processed = improve_transitions(processed)
    
    return processed
```

This syntax specification provides the complete formal grammar and semantic rules for the Turbulance language, enabling precise implementation of parsers and interpreters while maintaining the language's unique text-processing capabilities.
