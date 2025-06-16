//! Chart visualization components

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use super::{
    Visualization, VisualizationType, VisualizationData, VisualizationConfig,
    TimeSeriesPoint, CategoryData, ColorScheme
};

/// Chart-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfig {
    /// Chart title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Show grid lines
    pub show_grid: bool,
    /// Show legend
    pub show_legend: bool,
    /// Chart colors
    pub colors: Vec<String>,
    /// Line width for line charts
    pub line_width: f32,
    /// Point size for scatter plots
    pub point_size: f32,
    /// Bar width for bar charts
    pub bar_width: f32,
    /// Animation duration in milliseconds
    pub animation_duration: u32,
    /// Whether chart is interactive
    pub interactive: bool,
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            title: String::new(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            show_grid: true,
            show_legend: true,
            colors: vec![
                "#2563eb".to_string(), // Blue
                "#dc2626".to_string(), // Red
                "#16a34a".to_string(), // Green
                "#ca8a04".to_string(), // Yellow
                "#9333ea".to_string(), // Purple
                "#c2410c".to_string(), // Orange
            ],
            line_width: 2.0,
            point_size: 4.0,
            bar_width: 0.8,
            animation_duration: 750,
            interactive: true,
        }
    }
}

/// Text visualization struct (re-exported for compatibility)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextVisualization {
    pub id: String,
    pub title: String,
    pub chart_type: ChartType,
    pub config: ChartConfig,
}

/// Chart visualization data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartVisualizationData {
    /// Type of chart
    pub chart_type: ChartType,
    /// Data series
    pub series: Vec<DataSeries>,
    /// Axis configurations
    pub axes: AxisConfiguration,
    /// Annotations
    pub annotations: Vec<ChartAnnotation>,
    /// Statistical information
    pub statistics: Option<ChartStatistics>,
}

/// Types of charts supported
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart for continuous data
    Line,
    /// Bar chart for categorical data
    Bar,
    /// Horizontal bar chart
    HorizontalBar,
    /// Scatter plot for correlation analysis
    Scatter,
    /// Histogram for distribution analysis
    Histogram,
    /// Area chart for cumulative data
    Area,
    /// Stacked bar chart
    StackedBar,
    /// Multi-line chart
    MultiLine,
    /// Pie chart for proportional data
    Pie,
    /// Donut chart
    Donut,
}

/// Data series for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSeries {
    /// Series name
    pub name: String,
    /// Data points
    pub data: Vec<DataPoint>,
    /// Series color
    pub color: String,
    /// Series type (for mixed charts)
    pub series_type: Option<ChartType>,
    /// Whether this series is visible
    pub visible: bool,
    /// Series-specific styling
    pub style: SeriesStyle,
}

/// Individual data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// X-coordinate
    pub x: f64,
    /// Y-coordinate
    pub y: f64,
    /// Optional label
    pub label: Option<String>,
    /// Optional additional data
    pub metadata: HashMap<String, String>,
}

/// Styling for data series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesStyle {
    /// Line width
    pub line_width: f32,
    /// Point size
    pub point_size: f32,
    /// Fill opacity (0.0 - 1.0)
    pub fill_opacity: f32,
    /// Dash pattern for lines
    pub dash_pattern: Option<Vec<f32>>,
    /// Point shape
    pub point_shape: PointShape,
}

/// Point shapes for scatter plots
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PointShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Cross,
    Plus,
}

/// Axis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfiguration {
    /// X-axis configuration
    pub x_axis: AxisConfig,
    /// Y-axis configuration
    pub y_axis: AxisConfig,
}

/// Configuration for a single axis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    /// Axis label
    pub label: String,
    /// Minimum value
    pub min: Option<f64>,
    /// Maximum value
    pub max: Option<f64>,
    /// Tick interval
    pub tick_interval: Option<f64>,
    /// Number of ticks
    pub tick_count: Option<usize>,
    /// Axis type
    pub axis_type: AxisType,
    /// Show axis line
    pub show_line: bool,
    /// Show tick marks
    pub show_ticks: bool,
    /// Show grid lines
    pub show_grid: bool,
}

/// Types of axes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AxisType {
    /// Linear scale
    Linear,
    /// Logarithmic scale
    Logarithmic,
    /// Time scale
    Time,
    /// Categorical scale
    Categorical,
}

/// Chart annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartAnnotation {
    /// Annotation type
    pub annotation_type: AnnotationType,
    /// Position
    pub position: (f64, f64),
    /// Text content
    pub text: String,
    /// Styling
    pub style: AnnotationStyle,
}

/// Types of annotations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnnotationType {
    /// Text label
    Text,
    /// Trend line
    TrendLine,
    /// Horizontal line
    HorizontalLine,
    /// Vertical line
    VerticalLine,
    /// Rectangle
    Rectangle,
    /// Circle
    Circle,
    /// Arrow
    Arrow,
}

/// Annotation styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationStyle {
    /// Color
    pub color: String,
    /// Font size
    pub font_size: u32,
    /// Line width
    pub line_width: f32,
    /// Fill opacity
    pub fill_opacity: f32,
}

/// Statistical information about the chart
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartStatistics {
    /// Number of data points
    pub data_point_count: usize,
    /// Number of series
    pub series_count: usize,
    /// X-axis range
    pub x_range: (f64, f64),
    /// Y-axis range
    pub y_range: (f64, f64),
    /// Correlation coefficient (for scatter plots)
    pub correlation: Option<f64>,
    /// R-squared value (for trend analysis)
    pub r_squared: Option<f64>,
}

/// Main chart builder struct
pub struct ChartBuilder {
    config: ChartConfig,
    chart_type: ChartType,
    series: Vec<DataSeries>,
    axes: AxisConfiguration,
    annotations: Vec<ChartAnnotation>,
}

impl ChartBuilder {
    /// Create a new chart builder
    pub fn new(chart_type: ChartType) -> Self {
        Self {
            config: ChartConfig::default(),
            chart_type,
            series: Vec::new(),
            axes: AxisConfiguration {
                x_axis: AxisConfig::default(),
                y_axis: AxisConfig::default(),
            },
            annotations: Vec::new(),
        }
    }

    /// Set chart configuration
    pub fn with_config(mut self, config: ChartConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a data series
    pub fn add_series(mut self, series: DataSeries) -> Self {
        self.series.push(series);
        self
    }

    /// Add multiple series
    pub fn add_series_batch(mut self, series: Vec<DataSeries>) -> Self {
        self.series.extend(series);
        self
    }

    /// Set axis configuration
    pub fn with_axes(mut self, axes: AxisConfiguration) -> Self {
        self.axes = axes;
        self
    }

    /// Add annotation
    pub fn add_annotation(mut self, annotation: ChartAnnotation) -> Self {
        self.annotations.push(annotation);
        self
    }

    /// Build the visualization
    pub fn build(self) -> Result<Visualization> {
        let statistics = self.calculate_statistics();
        let chart_type = self.chart_type.clone(); // Clone before use
        
        let chart_data = ChartVisualizationData {
            chart_type: self.chart_type,
            series: self.series,
            axes: self.axes,
            annotations: self.annotations,
            statistics: Some(statistics),
        };

        let visualization = Visualization {
            id: uuid::Uuid::new_v4().to_string(),
            title: self.config.title.clone(),
            description: format!("{:?} chart visualization", chart_type),
            visualization_type: match chart_type {
                ChartType::Line | ChartType::MultiLine => VisualizationType::LineChart,
                ChartType::Bar | ChartType::HorizontalBar | ChartType::StackedBar => VisualizationType::BarChart,
                ChartType::Scatter => VisualizationType::ScatterPlot,
                ChartType::Histogram => VisualizationType::Histogram,
                _ => VisualizationType::LineChart,
            },
            data: VisualizationData::Custom(serde_json::to_value(chart_data)?),
            config: VisualizationConfig {
                width: 800,
                height: 600,
                color_scheme: ColorScheme::Default,
                show_legend: self.config.show_legend,
                show_grid: self.config.show_grid,
                font_size: 12,
                style_options: HashMap::new(),
            },
            metadata: HashMap::new(),
        };

        Ok(visualization)
    }

    /// Calculate statistics for the chart
    fn calculate_statistics(&self) -> ChartStatistics {
        let data_point_count: usize = self.series.iter().map(|s| s.data.len()).sum();
        let series_count = self.series.len();

        let all_x_values: Vec<f64> = self.series
            .iter()
            .flat_map(|s| s.data.iter().map(|p| p.x))
            .collect();
        
        let all_y_values: Vec<f64> = self.series
            .iter()
            .flat_map(|s| s.data.iter().map(|p| p.y))
            .collect();

        let x_range = if all_x_values.is_empty() {
            (0.0, 0.0)
        } else {
            (*all_x_values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
             *all_x_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
        };

        let y_range = if all_y_values.is_empty() {
            (0.0, 0.0)
        } else {
            (*all_y_values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
             *all_y_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
        };

        // Calculate correlation for scatter plots
        let correlation = if self.chart_type == ChartType::Scatter && !all_x_values.is_empty() {
            Some(calculate_correlation(&all_x_values, &all_y_values))
        } else {
            None
        };

        ChartStatistics {
            data_point_count,
            series_count,
            x_range,
            y_range,
            correlation,
            r_squared: correlation.map(|r| r * r),
        }
    }
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            label: String::new(),
            min: None,
            max: None,
            tick_interval: None,
            tick_count: None,
            axis_type: AxisType::Linear,
            show_line: true,
            show_ticks: true,
            show_grid: true,
        }
    }
}

impl Default for SeriesStyle {
    fn default() -> Self {
        Self {
            line_width: 2.0,
            point_size: 4.0,
            fill_opacity: 0.3,
            dash_pattern: None,
            point_shape: PointShape::Circle,
        }
    }
}

impl Default for AnnotationStyle {
    fn default() -> Self {
        Self {
            color: "#333333".to_string(),
            font_size: 12,
            line_width: 1.0,
            fill_opacity: 0.1,
        }
    }
}

/// Utility functions for chart creation
pub mod chart_utils {
    use super::*;

    /// Create a line chart from time series data
    pub fn create_line_chart(
        title: &str,
        data: Vec<TimeSeriesPoint>,
        x_label: &str,
        y_label: &str,
    ) -> Result<Visualization> {
        let series = DataSeries {
            name: "Data".to_string(),
            data: data.into_iter().map(|p| DataPoint {
                x: p.x,
                y: p.y,
                label: p.label,
                metadata: HashMap::new(),
            }).collect(),
            color: "#2563eb".to_string(),
            series_type: Some(ChartType::Line),
            visible: true,
            style: SeriesStyle::default(),
        };

        let mut config = ChartConfig::default();
        config.title = title.to_string();
        config.x_label = x_label.to_string();
        config.y_label = y_label.to_string();

        ChartBuilder::new(ChartType::Line)
            .with_config(config)
            .add_series(series)
            .build()
    }

    /// Create a bar chart from categorical data
    pub fn create_bar_chart(
        title: &str,
        data: Vec<CategoryData>,
        x_label: &str,
        y_label: &str,
    ) -> Result<Visualization> {
        let series = DataSeries {
            name: "Categories".to_string(),
            data: data.into_iter().enumerate().map(|(i, cat)| DataPoint {
                x: i as f64,
                y: cat.value,
                label: Some(cat.category),
                metadata: HashMap::new(),
            }).collect(),
            color: "#16a34a".to_string(),
            series_type: Some(ChartType::Bar),
            visible: true,
            style: SeriesStyle::default(),
        };

        let mut config = ChartConfig::default();
        config.title = title.to_string();
        config.x_label = x_label.to_string();
        config.y_label = y_label.to_string();

        let mut axes = AxisConfiguration {
            x_axis: AxisConfig::default(),
            y_axis: AxisConfig::default(),
        };
        axes.x_axis.axis_type = AxisType::Categorical;

        ChartBuilder::new(ChartType::Bar)
            .with_config(config)
            .add_series(series)
            .with_axes(axes)
            .build()
    }

    /// Create a scatter plot
    pub fn create_scatter_plot(
        title: &str,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        x_label: &str,
        y_label: &str,
    ) -> Result<Visualization> {
        if x_data.len() != y_data.len() {
            return Err(Error::visualization("X and Y data must have the same length"));
        }

        let data: Vec<DataPoint> = x_data.into_iter()
            .zip(y_data.into_iter())
            .map(|(x, y)| DataPoint {
                x,
                y,
                label: None,
                metadata: HashMap::new(),
            })
            .collect();

        let series = DataSeries {
            name: "Data Points".to_string(),
            data,
            color: "#dc2626".to_string(),
            series_type: Some(ChartType::Scatter),
            visible: true,
            style: SeriesStyle::default(),
        };

        let mut config = ChartConfig::default();
        config.title = title.to_string();
        config.x_label = x_label.to_string();
        config.y_label = y_label.to_string();

        ChartBuilder::new(ChartType::Scatter)
            .with_config(config)
            .add_series(series)
            .build()
    }

    /// Create a histogram from data values
    pub fn create_histogram(
        title: &str,
        data: Vec<f64>,
        bins: usize,
        x_label: &str,
    ) -> Result<Visualization> {
        let histogram_data = calculate_histogram(&data, bins);
        
        let series = DataSeries {
            name: "Frequency".to_string(),
            data: histogram_data.into_iter().enumerate().map(|(i, count)| DataPoint {
                x: i as f64,
                y: count as f64,
                label: None,
                metadata: HashMap::new(),
            }).collect(),
            color: "#ca8a04".to_string(),
            series_type: Some(ChartType::Histogram),
            visible: true,
            style: SeriesStyle::default(),
        };

        let mut config = ChartConfig::default();
        config.title = title.to_string();
        config.x_label = x_label.to_string();
        config.y_label = "Frequency".to_string();

        ChartBuilder::new(ChartType::Histogram)
            .with_config(config)
            .add_series(series)
            .build()
    }

    /// Add trend line to existing chart
    pub fn add_trend_line(
        visualization: &mut Visualization,
        x_data: &[f64],
        y_data: &[f64],
    ) -> Result<()> {
        if x_data.len() != y_data.len() || x_data.is_empty() {
            return Err(Error::visualization("Invalid data for trend line"));
        }

        let (slope, intercept) = calculate_linear_regression(x_data, y_data);
        
        let x_min = x_data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let x_max = x_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        let trend_annotation = ChartAnnotation {
            annotation_type: AnnotationType::TrendLine,
            position: (*x_min, slope * x_min + intercept),
            text: format!("y = {:.2}x + {:.2}", slope, intercept),
            style: AnnotationStyle {
                color: "#9333ea".to_string(),
                font_size: 10,
                line_width: 2.0,
                fill_opacity: 0.0,
            },
        };

        // Add to visualization metadata
        visualization.metadata.insert(
            "trend_line".to_string(),
            serde_json::to_string(&trend_annotation)
                .map_err(|e| Error::visualization(format!("Failed to serialize trend line: {}", e)))?
        );

        Ok(())
    }
}

/// Calculate correlation coefficient
fn calculate_correlation(x_data: &[f64], y_data: &[f64]) -> f64 {
    if x_data.len() != y_data.len() || x_data.is_empty() {
        return 0.0;
    }

    let n = x_data.len() as f64;
    let sum_x: f64 = x_data.iter().sum();
    let sum_y: f64 = y_data.iter().sum();
    let sum_xy: f64 = x_data.iter().zip(y_data.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = x_data.iter().map(|x| x * x).sum();
    let sum_y2: f64 = y_data.iter().map(|y| y * y).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Calculate linear regression
fn calculate_linear_regression(x_data: &[f64], y_data: &[f64]) -> (f64, f64) {
    if x_data.len() != y_data.len() || x_data.is_empty() {
        return (0.0, 0.0);
    }

    let n = x_data.len() as f64;
    let sum_x: f64 = x_data.iter().sum();
    let sum_y: f64 = y_data.iter().sum();
    let sum_xy: f64 = x_data.iter().zip(y_data.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = x_data.iter().map(|x| x * x).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    (slope, intercept)
}

/// Calculate histogram bins
fn calculate_histogram(data: &[f64], bins: usize) -> Vec<usize> {
    if data.is_empty() || bins == 0 {
        return vec![0; bins];
    }

    let min_val = *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_val = *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    if min_val == max_val {
        let mut result = vec![0; bins];
        if bins > 0 {
            result[0] = data.len();
        }
        return result;
    }

    let bin_width = (max_val - min_val) / bins as f64;
    let mut histogram = vec![0; bins];

    for &value in data {
        let bin_index = ((value - min_val) / bin_width).floor() as usize;
        let bin_index = bin_index.min(bins - 1); // Handle edge case where value == max_val
        histogram[bin_index] += 1;
    }

    histogram
} 