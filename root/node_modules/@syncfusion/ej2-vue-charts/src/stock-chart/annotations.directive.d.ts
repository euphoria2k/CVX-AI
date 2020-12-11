import Vue from 'vue';
export declare class StockChartAnnotationsDirective extends Vue {
    render(): void;
}
export declare const StockChartAnnotationsPlugin: {
    name: string;
    install(Vue: any): void;
};
/**
 * `e-annotation` directive represent a annotation of the VueJS Chart.
 * It must be contained in a Chart component(`ejs-chart`).
 * ```vue
 * <ejs-stockchart>
 *   <e-stockchart-annotations>
 *    <e-annotation content='ID' />
 *    <e-annotation content='ID' />
 *   </e-annotations>
 * </ejs-chart>
 * ```
 */
export declare class StockChartAnnotationDirective extends Vue {
    render(): void;
}
export declare const StockChartAnnotationPlugin: {
    name: string;
    install(Vue: any): void;
};
